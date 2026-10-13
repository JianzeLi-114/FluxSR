# Adapted from the Diffusers FLUX transformer forward (Apache-2.0).
# FluxSR modification: compute the image-token ADL inside each block checkpoint.
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)

from utils.others import attention_diversification_loss


def _block_forward_with_adl(
    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb,
    joint_attention_kwargs,
):
    """Reduce ADL inside the checkpoint region instead of retaining all features."""
    encoder_hidden_states, hidden_states = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    return encoder_hidden_states, hidden_states, attention_diversification_loss(hidden_states)


def transformer_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    compute_adl: bool = False,
):
    """
    Diffusers FLUX forward with optional all-layer image-token ADL.

    Uses the separate text/image block API in Diffusers 0.37. The second
    returned item is a scalar ADL, or None when compute_adl is false.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, image_tokens, channels)`):
            Packed image latents.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tensor as the first item.
        compute_adl (`bool`, *optional*, defaults to `False`):
            Average Eq. (23) over image tokens in every dual and single block.

    Returns:
        A pair of (model output, ADL). The first item is Transformer2DModelOutput
        when return_dict is True and a tensor otherwise. ADL is None when disabled.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        # logger.warning(
        #     "Passing `txt_ids` 3d torch.Tensor is deprecated."
        #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
        # )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        # logger.warning(
        #     "Passing `img_ids` 3d torch.Tensor is deprecated."
        #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
        # )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    adl_sum = None
    adl_layers = 0

    def run_block(block, hidden_states, encoder_hidden_states):
        block_forward = partial(_block_forward_with_adl, block) if compute_adl else block
        inputs = (
            hidden_states, encoder_hidden_states, temb,
            image_rotary_emb, joint_attention_kwargs,
        )
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return self._gradient_checkpointing_func(block_forward, *inputs)
        return block_forward(*inputs)

    for index_block, block in enumerate(self.transformer_blocks):
        block_output = run_block(block, hidden_states, encoder_hidden_states)
        encoder_hidden_states, hidden_states = block_output[:2]
        if compute_adl:
            adl_sum = block_output[2] if adl_sum is None else adl_sum + block_output[2]
            adl_layers += 1

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    for index_block, block in enumerate(self.single_transformer_blocks):
        block_output = run_block(block, hidden_states, encoder_hidden_states)
        encoder_hidden_states, hidden_states = block_output[:2]
        if compute_adl:
            adl_sum = block_output[2] if adl_sum is None else adl_sum + block_output[2]
            adl_layers += 1

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

    adl_loss = adl_sum / adl_layers if adl_layers else None

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output, adl_loss)

    return Transformer2DModelOutput(sample=output), adl_loss
