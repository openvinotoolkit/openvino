# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import inspect
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
from tempfile import TemporaryDirectory
import PIL
import numpy as np
import torch
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.utils import randn_tensor, PIL_INTERPOLATION
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from optimum.intel.openvino import OVModelForCausalLM
from openvino.runtime import Model, Core, Tensor, Type
from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class OVMPTModel(OVModelForCausalLM):
    def __init__(
        self,
        model: Model,
        config: PretrainedConfig = None,
        device: str = 'CPU',
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf['mpt'] = NormalizedTextConfig.with_args(num_layers='n_layers', num_attention_heads='n_heads')
        super().__init__(model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs)

    def _reshape(
        self,
        model: Model,
        *args,
        **kwargs,
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if shapes[inputs].rank.get_length() in [2, 3]:
                shapes[inputs][1] = -1
            else:
                if '.key' in inputs.get_any_name():
                    shapes[inputs][3] = -1
                else:
                    shapes[inputs][2] = -1
        model.reshape(shapes)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            num_attention_heads = (
                self.normalized_config.num_attention_heads if self.config.model_type == 'bloom' else 1
            )
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                if shape.rank.get_length() == 4 and shape[3].is_dynamic:
                    shape[3] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs['input_ids'] = np.array(input_ids)

        # Add the attention_mask inputs when needed
        if 'attention_mask' in self.input_names and attention_mask is not None:
            inputs['attention_mask'] = np.array(attention_mask)

        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor('logits').data).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i:i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


class OVFalconModel(OVModelForCausalLM):
    def __init__(
        self,
        model: Model,
        config: PretrainedConfig = None,
        device: str = 'CPU',
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf['RefinedWebModel'] = NormalizedTextConfig.with_args(num_layers='n_layer', num_attention_heads='n_head')
        NormalizedConfigManager._conf['falcon'] = NormalizedTextConfig.with_args(num_layers='num_hidden_layers', num_attention_heads='num_attention_heads')
        NormalizedConfigManager._conf['RefinedWeb'] = NormalizedTextConfig.with_args(num_layers='n_layer', num_attention_heads='n_head')
        super().__init__(model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs)

    def _reshape(
        self,
        model: Model,
        *args,
        **kwargs,
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if shapes[inputs].rank.get_length() in [2, 4]:
                shapes[inputs][0] = -1
            if shapes[inputs].rank.get_length() in [2, 3]:
                shapes[inputs][1] = -1
            if shapes[inputs].rank.get_length() == 4:
                shapes[inputs][2] = -1
        model.reshape(shapes)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            num_attention_heads = (
                self.normalized_config.num_attention_heads if self.config.model_type == 'bloom' else 1
            )
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads if shape[0].is_dynamic else shape[0]
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs['input_ids'] = np.array(input_ids)

        # Add the attention_mask inputs when needed
        if 'attention_mask' in self.input_names and attention_mask is not None:
            inputs['attention_mask'] = np.array(attention_mask)

        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor('logits').data).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i:i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


class OVLDMSuperResolutionPipeline(DiffusionPipeline):

    def __init__(self, model_path: Path, core: Core, device: str):
        super().__init__()
        self.vqvae = core.compile_model(model_path / 'vqvae.xml', device)
        self.unet = core.compile_model(model_path / 'unet.xml', device)
        self.scheduler = LMSDiscreteScheduler.from_config(model_path / 'scheduler_config.json')
        self._unet_output = self.unet.output(0)
        self._vqvae_output = self.vqvae.output(0)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        batch_size: Optional[int] = 1,
        num_inference_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        image = image

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError(f'`image` has to be of type `PIL.Image.Image` or `torch.Tensor` but is {type(image)}')

        if isinstance(image, PIL.Image.Image):
            image = self.preprocess(image)

        height, width = image.shape[-2:]

        # in_channels should be 6: 3 for latents, 3 for low resolution image
        latents_shape = (batch_size, 3, height, width)
        latents = randn_tensor(latents_shape, generator=generator)
        # set timesteps and move to the correct device
        # self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        latents = latents.numpy()
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs['eta'] = eta

        for t in self.progress_bar(timesteps_tensor):
            # concat latents and low resolution image in the channel dimension.
            latents_input = np.concatenate([latents, image], axis=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            # predict the noise residual
            noise_pred = self.unet([latents_input, t])[self._unet_output]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents))['prev_sample'].numpy()

        # decode the image latents with the VQVAE
        image = self.vqvae(latents)[self._vqvae_output]
        image = image / 2 + 0.5
        image = image.transpose(0, 2, 3, 1)

        if output_type == 'pil':
            image = self.numpy_to_pil(image)
        return image

    @staticmethod
    def preprocess(image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL_INTERPOLATION['lanczos'])
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0


class OVChatGLM2Model(OVModelForCausalLM):
    def __init__(
        self,
        model: Model,
        config: PretrainedConfig = None,
        device: str = 'CPU',
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf['chatglm'] = NormalizedTextConfig.with_args(num_layers='num_layers', num_attention_heads='num_attention_heads')
        super().__init__(model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs)

    def _reshape(
        self,
        model: Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        batch_size, seq_length = input_ids.shape
        position_ids = self.get_position_ids(input_ids, 'cpu')

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            position_ids = position_ids[..., -1:] + 1
            past = past_key_values if past_key_values is not None else past

        return {
                'input_ids': input_ids,
                'past_key_values': past,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'use_cache': self.use_cache,
                'token_type_ids': None,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        if past_key_values is not None:
            if self._pkv_precision == Type.bf16:
                # numpy does not support bf16, pretending f16, should change to bf16
                past_key_values = tuple(
                    Tensor(past_key_value, past_key_value.shape, Type.bf16)
                    for pkv_per_layer in past_key_values
                    for past_key_value in pkv_per_layer
                )
            else:
                # Flatten the past_key_values
                past_key_values = tuple(
                    past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                )
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            # shape_input_ids = input_ids.shape
            # num_attention_heads = (
            #     self.normalized_config.num_attention_heads if self.config.model_type == 'bloom' else 1
            # )
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = 0
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs['input_ids'] = np.array(input_ids)

        if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
            inputs['position_ids'] = np.array(kwargs['position_ids'])

        # Add the attention_mask inputs when needed
        if 'attention_mask' in self.input_names and attention_mask is not None:
            inputs['attention_mask'] = np.array(attention_mask)

        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor('logits').data).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i:i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids


class OVChatGLMModel(OVModelForCausalLM):
    position_encoding_2d = True
    num_layers = 28
    max_sequence_length = 128
    bos_token_id = 130004
    eos_token_id = 130005
    mask_token_id = 130000
    gmask_token_id = 130001

    def __init__(
        self,
        model: Model,
        config: PretrainedConfig = None,
        device: str = 'CPU',
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf['chatglm'] = NormalizedTextConfig.with_args(num_layers='num_layers', num_attention_heads='num_attention_heads')
        super().__init__(model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs)
        self.key_value_input_names = ['past_key_values']
        self.key_value_output_names = [o.any_name for o in self.model.outputs[1:]]

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        batch_size, seq_length = input_ids.shape
        mask = self.mask_token_id
        g_mask = self.gmask_token_id
        seqs = input_ids.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            tmp_mask_token = g_mask if g_mask in seq else mask
            use_gmask = tmp_mask_token == g_mask
            mask_positions.append(seq.index(tmp_mask_token))
            use_gmasks.append(use_gmask)

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            # Next Step Inference
            last_token = input_ids[:, -1].unsqueeze(-1)
            # if attention_mask is not None:
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.bos_token_id) for seq in seqs]
                if self.position_encoding_2d:    # position_encoding_2d = True
                    position_ids = torch.tensor(
                        [[mask_position, seq_length - context_length] for mask_position, context_length in
                            zip(mask_positions, context_lengths)], dtype=torch.long, device=input_ids.device).unsqueeze(-1)
                else:
                    position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long,
                                                device=input_ids.device).unsqueeze(-1)

            if past is None:
                past = self.get_past_key_values(past_key_values)
            return {
                'input_ids': last_token,
                'past_key_values': past,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'use_cache': self.use_cache,
                'token_type_ids': None
            }
        else:
            # First Step Inference
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                attention_mask = None
            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device,
                )
            if position_ids is None:
                position_ids = self.get_position_ids(
                    input_ids,
                    device=input_ids.device,
                    mask_positions=mask_positions,
                    use_gmasks=use_gmasks,
                )
            past_key_values = None
            if self.use_cache:
                past_key_values = np.zeros((self.num_layers, 2, 0, 1, 32, 128))
                # numpy does not support bf16, pretending f16, should change to bf16
                if self._pkv_precision == Type.bf16:
                    past_key_values = Tensor(past_key_values, past_key_values.shape, Type.bf16)
            return {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'past_key_values': past_key_values,
                'use_cache': self.use_cache,
                'token_type_ids': None,
            }

    def get_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, device, use_gmasks=None):
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        if self.position_encoding_2d:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [torch.cat((
                torch.zeros(context_length, dtype=torch.long, device=device),
                torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1,
            )) for context_length in context_lengths]
            block_position_ids = torch.stack(block_position_ids, dim=0)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]

        return position_ids

    @staticmethod
    def get_past_key_values(pkv):
        pkv_combined = []
        for i in range(0, len(pkv)):
            past_key_values_pair = np.stack(pkv[i], axis=0)
            pkv_combined.append(past_key_values_pair)
        pkv_combined = np.array(pkv_combined)
        return pkv_combined

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        inputs = {}
        if past_key_values is not None:
            inputs['past_key_values'] = past_key_values
        inputs['input_ids'] = np.array(input_ids)

        # Add the attention_mask inputs when needed
        if 'attention_mask' in self.input_names and attention_mask is not None:
            inputs['attention_mask'] = np.array(attention_mask)

        if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
            inputs['position_ids'] = np.array(kwargs['position_ids'])

        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor('logits').data).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i:i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _reshape(
        self,
        model: Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        return model
