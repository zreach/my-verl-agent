# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _forward_micro_batch_response_logits(self, micro_batch, temperature) -> torch.Tensor:
        """Return response-position logits for full/top-k OPD."""
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)
                if self.use_ulysses_sp:
                    logits_rmpad = gather_outpus_and_unpad(
                        logits_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                full_logits = pad_input(
                    hidden_states=logits_rmpad,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                logits = full_logits[:, -response_length - 1 : -1, :]
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits = output.logits[:, -response_length - 1 : -1, :]
                logits.div_(temperature)

        return logits

    def _resolve_distillation_mode(self, distillation_config):
        method = distillation_config.get("method", None)
        target = distillation_config.get("target", "sampled")
        loss_mode = distillation_config.get("loss_mode", "k3")

        if method is None:
            if loss_mode == "forward_kl_topk":
                method = "gkd"
                target = "topk"
            else:
                method = "pg" if distillation_config.get("use_policy_gradient", True) else "gkd"

        if method == "pg" and target != "sampled":
            raise ValueError("OPD PG follows the sampled-token estimator and only supports distillation.target=sampled.")
        if method == "vopd" and target not in ["full", "topk"]:
            raise ValueError("distillation.method=vopd requires distillation.target=full or topk")
        return method, target

    def _opd_token_kl_from_teacher_distribution(self, student_logits, teacher_log_probs, response_mask, topk_indices=None):
        student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
        if topk_indices is not None:
            student_log_probs = torch.gather(student_log_probs, dim=-1, index=topk_indices)
        teacher_log_probs = teacher_log_probs.float()
        teacher_log_probs = teacher_log_probs - torch.logsumexp(teacher_log_probs, dim=-1, keepdim=True)
        teacher_probs = teacher_log_probs.exp()
        token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        token_ce = -(teacher_probs * student_log_probs).sum(dim=-1)
        token_kl = token_kl * response_mask
        token_ce = token_ce * response_mask
        return token_kl, token_ce

    def _opd_forward_kl_from_teacher_distribution(self, student_logits, teacher_log_probs, response_mask, topk_indices=None):
        """my-verl style forward KL over teacher support without renormalizing top-k mass."""
        student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
        if topk_indices is not None:
            student_log_probs = torch.gather(student_log_probs, dim=-1, index=topk_indices)
        teacher_log_probs = teacher_log_probs.float()
        teacher_probs = teacher_log_probs.exp()
        token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        token_kl = token_kl.clamp_min(0.0)
        token_kl = token_kl * response_mask
        return token_kl, token_kl

    def _opd_reverse_kl_baseline(self, student_logits, teacher_log_probs, response_mask, topk_indices=None):
        student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
        if topk_indices is not None:
            student_log_probs = torch.gather(student_log_probs, dim=-1, index=topk_indices)
            student_log_probs = student_log_probs - torch.logsumexp(student_log_probs, dim=-1, keepdim=True)
        teacher_log_probs = teacher_log_probs.float()
        teacher_log_probs = teacher_log_probs - torch.logsumexp(teacher_log_probs, dim=-1, keepdim=True)
        student_probs = student_log_probs.exp()
        reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        return reverse_kl * response_mask

    def _opd_reverse_kl_from_log_probs(self, student_log_probs, teacher_log_probs, response_mask):
        student_log_probs = student_log_probs.float()
        teacher_log_probs = teacher_log_probs.float()
        student_log_probs = student_log_probs - torch.logsumexp(student_log_probs, dim=-1, keepdim=True)
        teacher_log_probs = teacher_log_probs - torch.logsumexp(teacher_log_probs, dim=-1, keepdim=True)
        student_probs = student_log_probs.exp()
        reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        return reverse_kl * response_mask

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_opd_distribution(self, data: DataProto, target: str = "topk", topk: int = 32, normalize_topk: bool = False) -> DataProto:
        """Compute teacher full-vocab or top-k log-prob distributions for OPD."""
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if target == "topk" and "student_topk_indices" in data.batch.keys():
            select_keys.append("student_topk_indices")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        topk_log_probs_lst = []
        topk_indices_lst = []
        full_log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                logits = self._forward_micro_batch_response_logits(micro_batch, temperature=temperature)
                if target == "full":
                    full_log_probs_lst.append(torch.log_softmax(logits.float(), dim=-1))
                elif target == "topk":
                    if "student_topk_indices" in micro_batch:
                        topk_indices = micro_batch["student_topk_indices"]
                        gathered_logits = torch.gather(logits.float(), dim=-1, index=topk_indices)
                        if normalize_topk:
                            topk_log_probs_lst.append(torch.log_softmax(gathered_logits, dim=-1))
                        else:
                            topk_log_probs = torch.gather(torch.log_softmax(logits.float(), dim=-1), dim=-1, index=topk_indices)
                            topk_log_probs_lst.append(topk_log_probs)
                        topk_indices_lst.append(topk_indices)
                    else:
                        values, topk_indices = torch.topk(logits.float(), k=min(topk, logits.size(-1)), dim=-1)
                        if normalize_topk:
                            topk_log_probs_lst.append(torch.log_softmax(values, dim=-1))
                        else:
                            topk_log_probs = torch.gather(torch.log_softmax(logits.float(), dim=-1), dim=-1, index=topk_indices)
                            topk_log_probs_lst.append(topk_log_probs)
                        topk_indices_lst.append(topk_indices)
                else:
                    raise ValueError(f"Unsupported OPD distribution target: {target}")

        if target == "full":
            full_log_probs = torch.concat(full_log_probs_lst, dim=0)
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=full_log_probs.device)
                full_log_probs = full_log_probs[revert_indices]
            return DataProto.from_dict(tensors={"teacher_full_log_probs": full_log_probs})

        topk_log_probs = torch.concat(topk_log_probs_lst, dim=0)
        topk_indices = torch.concat(topk_indices_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=topk_log_probs.device)
            topk_log_probs = topk_log_probs[revert_indices]
            topk_indices = topk_indices[revert_indices]
        return DataProto.from_dict(tensors={"teacher_topk_log_probs": topk_log_probs, "teacher_topk_indices": topk_indices})

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_opd_topk_indices(self, data: DataProto, topk: int = 32) -> DataProto:
        """Compute student top-k token ids at response positions for vOPD top-k baseline."""
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        topk_indices_lst = []
        topk_log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                logits = self._forward_micro_batch_response_logits(micro_batch, temperature=temperature)
                topk_logits, topk_indices = torch.topk(logits.float(), k=min(topk, logits.size(-1)), dim=-1)
                topk_indices_lst.append(topk_indices)
                topk_log_probs_lst.append(torch.log_softmax(topk_logits, dim=-1))

        topk_indices = torch.concat(topk_indices_lst, dim=0)
        topk_log_probs = torch.concat(topk_log_probs_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=topk_indices.device)
            topk_indices = topk_indices[revert_indices]
            topk_log_probs = topk_log_probs[revert_indices]
        return DataProto.from_dict(tensors={"student_topk_indices": topk_indices, "student_topk_log_probs": topk_log_probs})

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        distillation_config = self.config.get("distillation", {})
        distillation_enabled = distillation_config.get("enabled", False)
        if distillation_enabled:
            distillation_method, distillation_target = self._resolve_distillation_mode(distillation_config)
            if distillation_method == "vopd":
                select_keys.append("teacher_log_probs")
                if distillation_target == "full":
                    select_keys.append("teacher_full_log_probs")
                elif distillation_target == "topk":
                    select_keys.extend(["teacher_topk_log_probs", "teacher_topk_indices", "student_topk_log_probs"])
            elif distillation_target == "sampled":
                select_keys.append("teacher_log_probs")
            elif distillation_method == "gkd" and distillation_target == "full":
                select_keys.append("teacher_full_log_probs")
            elif distillation_method == "gkd" and distillation_target == "topk":
                select_keys.extend(["teacher_topk_log_probs", "teacher_topk_indices"])
            else:
                raise ValueError(f"Unsupported OPD config: method={distillation_method}, target={distillation_target}")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if distillation_enabled:
                        distillation_method, distillation_target = self._resolve_distillation_mode(distillation_config)

                        if distillation_method == "vopd":
                            teacher_log_probs = data["teacher_log_probs"]
                            sample_reward = teacher_log_probs - log_prob
                            if distillation_target == "full":
                                student_logits = self._forward_micro_batch_response_logits(data, temperature=temperature)
                                baseline_kl = self._opd_reverse_kl_baseline(
                                    student_logits=student_logits,
                                    teacher_log_probs=data["teacher_full_log_probs"],
                                    response_mask=response_mask,
                                )
                            else:
                                baseline_kl = self._opd_reverse_kl_from_log_probs(
                                    student_log_probs=data["student_topk_log_probs"],
                                    teacher_log_probs=data["teacher_topk_log_probs"],
                                    response_mask=response_mask,
                                )
                            distillation_losses = -(sample_reward + baseline_kl.detach()) * response_mask
                            distill_direct_losses = distillation_losses
                        elif distillation_target == "sampled":
                            teacher_log_probs = data["teacher_log_probs"]
                            distill_loss_mode = distillation_config.get("loss_mode", "k3")
                            distillation_losses = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=teacher_log_probs,
                                kl_penalty=distill_loss_mode,
                            )
                            distill_direct_losses = distillation_losses
                        else:
                            student_logits = self._forward_micro_batch_response_logits(data, temperature=temperature)
                            if distillation_target == "full":
                                teacher_distribution = data["teacher_full_log_probs"]
                                distillation_losses, distill_direct_losses = self._opd_forward_kl_from_teacher_distribution(
                                    student_logits=student_logits,
                                    teacher_log_probs=teacher_distribution,
                                    response_mask=response_mask,
                                )
                            else:
                                teacher_distribution = data["teacher_topk_log_probs"]
                                teacher_indices = data["teacher_topk_indices"]
                                distillation_losses, distill_direct_losses = self._opd_forward_kl_from_teacher_distribution(
                                    student_logits=student_logits,
                                    teacher_log_probs=teacher_distribution,
                                    response_mask=response_mask,
                                    topk_indices=teacher_indices,
                                )
                        loss_max_clamp = distillation_config.get("loss_max_clamp", None)
                        if loss_max_clamp is not None:
                            distillation_losses = distillation_losses.clamp(
                                min=-loss_max_clamp,
                                max=loss_max_clamp,
                            )
                            if distillation_method == "gkd" or distillation_target == "sampled":
                                distill_direct_losses = distillation_losses

                        if distillation_method in ["pg", "vopd"]:
                            distill_pg_loss, distill_pg_clipfrac, distill_ppo_kl, distill_pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=-distillation_losses.detach(),
                                response_mask=response_mask,
                                cliprange=distillation_config.get("clip_ratio", clip_ratio),
                                cliprange_low=distillation_config.get("clip_ratio_low", clip_ratio_low),
                                cliprange_high=distillation_config.get("clip_ratio_high", clip_ratio_high),
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                            distill_loss = distill_pg_loss
                            metrics["actor/distillation_pg_clipfrac"] = distill_pg_clipfrac.detach().item()
                            metrics["actor/distillation_ppo_kl"] = distill_ppo_kl.detach().item()
                            metrics["actor/distillation_pg_clipfrac_lower"] = distill_pg_clipfrac_lower.detach().item()
                        elif distillation_method == "gkd":
                            distill_loss = agg_loss(
                                loss_mat=distill_direct_losses,
                                loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                        else:
                            raise ValueError(f"Unsupported distillation.method: {distillation_method}")

                        if not distillation_config.get("use_task_rewards", True):
                            policy_loss = torch.zeros_like(policy_loss)
                        distill_coef = distillation_config.get("distillation_loss_coef", 1.0)
                        policy_loss = policy_loss + distill_coef * distill_loss
                        metrics["actor/distillation_loss"] = distill_loss.detach().item()
                        metrics["actor/distillation_abs_loss"] = (
                            torch.masked_select(distillation_losses.detach().abs(), response_mask.bool()).mean().item()
                        )
                        metrics["actor/distillation_loss_coef"] = distill_coef
                        metrics["actor/distillation_method_pg"] = float(distillation_method == "pg")
                        metrics["actor/distillation_method_vopd"] = float(distillation_method == "vopd")
                        metrics["actor/distillation_target_full"] = float(distillation_target == "full")
                        metrics["actor/distillation_target_topk"] = float(distillation_target == "topk")

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
