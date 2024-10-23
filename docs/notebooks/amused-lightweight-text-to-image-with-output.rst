Lightweight image generation with aMUSEd and OpenVINO
=====================================================

`Amused <https://huggingface.co/docs/diffusers/api/pipelines/amused>`__
is a lightweight text to image model based off of the
`muse <https://arxiv.org/pdf/2301.00704.pdf>`__ architecture. Amused is
particularly useful in applications that require a lightweight and fast
model such as generating many images quickly at once.

Amused is a VQVAE token based transformer that can generate an image in
fewer forward passes than many diffusion models. In contrast with muse,
it uses the smaller text encoder CLIP-L/14 instead of t5-xxl. Due to its
small parameter count and few forward pass generation process, amused
can generate many images quickly. This benefit is seen particularly at
larger batch sizes.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Load and run the original
   pipeline <#load-and-run-the-original-pipeline>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__

   -  `Convert the Text Encoder <#convert-the-text-encoder>`__
   -  `Convert the U-ViT transformer <#convert-the-u-vit-transformer>`__
   -  `Convert VQ-GAN decoder
      (VQVAE) <#convert-vq-gan-decoder-vqvae>`__

-  `Compiling models and prepare
   pipeline <#compiling-models-and-prepare-pipeline>`__
-  `Quantization <#quantization>`__

   -  `Prepare calibration dataset <#prepare-calibration-dataset>`__
   -  `Run model quantization <#run-model-quantization>`__
   -  `Compute Inception Scores and inference
      time <#compute-inception-scores-and-inference-time>`__

-  `Interactive inference <#interactive-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    %pip install -q transformers "diffusers>=0.25.0" "openvino>=2023.2.0" "accelerate>=0.20.3" "gradio>=4.19" "torch>=2.1" "pillow" "torchmetrics" "torch-fidelity" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "nncf>=2.9.0" datasets


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    # Fetch the notebook utils script from the openvino_notebooks repo
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)




.. parsed-literal::

    24692



Load and run the original pipeline
----------------------------------



.. code:: ipython3

    import torch
    from diffusers import AmusedPipeline
    
    
    pipe = AmusedPipeline.from_pretrained(
        "amused/amused-256",
    )
    
    prompt = "kind smiling ghost"
    image = pipe(prompt, generator=torch.Generator("cpu").manual_seed(8)).images[0]
    image.save("text2image_256.png")



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/12 [00:00<?, ?it/s]


.. code:: ipython3

    image




.. image:: amused-lightweight-text-to-image-with-output_files/amused-lightweight-text-to-image-with-output_7_0.png



Convert the model to OpenVINO IR
--------------------------------



aMUSEd consists of three separately trained components: a pre-trained
CLIP-L/14 text encoder, a VQ-GAN, and a U-ViT.

.. figure:: https://cdn-uploads.huggingface.co/production/uploads/5dfcb1aada6d0311fd3d5448/97ca2Vqm7jBfCAzq20TtF.png
   :alt: image_png

   image_png

During inference, the U-ViT is conditioned on the text encoder’s hidden
states and iteratively predicts values for all masked tokens. The cosine
masking schedule determines a percentage of the most confident token
predictions to be fixed after every iteration. After 12 iterations, all
tokens have been predicted and are decoded by the VQ-GAN into image
pixels.

Define paths for converted models:

.. code:: ipython3

    from pathlib import Path
    
    
    TRANSFORMER_OV_PATH = Path("models/transformer_ir.xml")
    TEXT_ENCODER_OV_PATH = Path("models/text_encoder_ir.xml")
    VQVAE_OV_PATH = Path("models/vqvae_ir.xml")

Define the conversion function for PyTorch modules. We use
``ov.convert_model`` function to obtain OpenVINO Intermediate
Representation object and ``ov.save_model`` function to save it as XML
file.

.. code:: ipython3

    import torch
    
    import openvino as ov
    
    
    def convert(model: torch.nn.Module, xml_path: str, example_input):
        xml_path = Path(xml_path)
        if not xml_path.exists():
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                converted_model = ov.convert_model(model, example_input=example_input)
            ov.save_model(converted_model, xml_path, compress_to_fp16=False)
    
            # cleanup memory
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()

Convert the Text Encoder
~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, text_encoder):
            super().__init__()
            self.text_encoder = text_encoder
    
        def forward(self, input_ids=None, return_dict=None, output_hidden_states=None):
            outputs = self.text_encoder(
                input_ids=input_ids,
                return_dict=return_dict,
                output_hidden_states=output_hidden_states,
            )
    
            return outputs.text_embeds, outputs.last_hidden_state, outputs.hidden_states
    
    
    input_ids = pipe.tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
    )
    
    input_example = {
        "input_ids": input_ids.input_ids,
        "return_dict": torch.tensor(True),
        "output_hidden_states": torch.tensor(True),
    }
    
    convert(TextEncoderWrapper(pipe.text_encoder), TEXT_ENCODER_OV_PATH, input_example)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4779: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:88: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if input_shape[-1] > 1 or self.sliding_window is not None:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:164: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:808: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      encoder_states = () if output_hidden_states else None
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:813: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if output_hidden_states:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:836: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if output_hidden_states:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:839: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not return_dict:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:935: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not return_dict:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:1426: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not return_dict:


Convert the U-ViT transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    class TransformerWrapper(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer
    
        def forward(
            self,
            latents=None,
            micro_conds=None,
            pooled_text_emb=None,
            encoder_hidden_states=None,
        ):
            return self.transformer(
                latents,
                micro_conds=micro_conds,
                pooled_text_emb=pooled_text_emb,
                encoder_hidden_states=encoder_hidden_states,
            )
    
    
    shape = (1, 16, 16)
    latents = torch.full(shape, pipe.scheduler.config.mask_token_id, dtype=torch.long)
    latents = torch.cat([latents] * 2)
    
    
    example_input = {
        "latents": latents,
        "micro_conds": torch.rand([2, 5], dtype=torch.float32),
        "pooled_text_emb": torch.rand([2, 768], dtype=torch.float32),
        "encoder_hidden_states": torch.rand([2, 77, 768], dtype=torch.float32),
    }
    
    
    pipe.transformer.eval()
    w_transformer = TransformerWrapper(pipe.transformer)
    convert(w_transformer, TRANSFORMER_OV_PATH, example_input)

Convert VQ-GAN decoder (VQVAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function ``get_latents`` is
needed to return real latents for the conversion. Due to the VQVAE
implementation autogenerated tensor of the required shape is not
suitable. This function repeats part of ``AmusedPipeline``.

.. code:: ipython3

    def get_latents():
        shape = (1, 16, 16)
        latents = torch.full(shape, pipe.scheduler.config.mask_token_id, dtype=torch.long)
        model_input = torch.cat([latents] * 2)
    
        model_output = pipe.transformer(
            model_input,
            micro_conds=torch.rand([2, 5], dtype=torch.float32),
            pooled_text_emb=torch.rand([2, 768], dtype=torch.float32),
            encoder_hidden_states=torch.rand([2, 77, 768], dtype=torch.float32),
        )
        guidance_scale = 10.0
        uncond_logits, cond_logits = model_output.chunk(2)
        model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
    
        latents = pipe.scheduler.step(
            model_output=model_output,
            timestep=torch.tensor(0),
            sample=latents,
        ).prev_sample
    
        return latents
    
    
    class VQVAEWrapper(torch.nn.Module):
        def __init__(self, vqvae):
            super().__init__()
            self.vqvae = vqvae
    
        def forward(self, latents=None, force_not_quantize=True, shape=None):
            outputs = self.vqvae.decode(
                latents,
                force_not_quantize=force_not_quantize,
                shape=shape.tolist(),
            )
    
            return outputs
    
    
    latents = get_latents()
    example_vqvae_input = {
        "latents": latents,
        "force_not_quantize": torch.tensor(True),
        "shape": torch.tensor((1, 16, 16, 64)),
    }
    
    convert(VQVAEWrapper(pipe.vqvae), VQVAE_OV_PATH, example_vqvae_input)


.. parsed-literal::

    /tmp/ipykernel_2578393/3779428577.py:34: TracerWarning: Converting a tensor to a Python list might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      shape=shape.tolist(),
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/autoencoders/vq_model.py:144: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not force_not_quantize:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:147: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:162: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if hidden_states.shape[0] >= 64:


Compiling models and prepare pipeline
-------------------------------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    core = ov.Core()
    
    ov_text_encoder = core.compile_model(TEXT_ENCODER_OV_PATH, device.value)
    ov_transformer = core.compile_model(TRANSFORMER_OV_PATH, device.value)
    ov_vqvae = core.compile_model(VQVAE_OV_PATH, device.value)

Let’s create callable wrapper classes for compiled models to allow
interaction with original ``AmusedPipeline`` class. Note that all of
wrapper classes return ``torch.Tensor``\ s instead of ``np.array``\ s.

.. code:: ipython3

    from collections import namedtuple
    
    
    class ConvTextEncoderWrapper(torch.nn.Module):
        def __init__(self, text_encoder, config):
            super().__init__()
            self.config = config
            self.text_encoder = text_encoder
    
        def forward(self, input_ids=None, return_dict=None, output_hidden_states=None):
            inputs = {
                "input_ids": input_ids,
                "return_dict": return_dict,
                "output_hidden_states": output_hidden_states,
            }
    
            outs = self.text_encoder(inputs)
    
            outputs = namedtuple("CLIPTextModelOutput", ("text_embeds", "last_hidden_state", "hidden_states"))
    
            text_embeds = torch.from_numpy(outs[0])
            last_hidden_state = torch.from_numpy(outs[1])
            hidden_states = list(torch.from_numpy(out) for out in outs.values())[2:]
    
            return outputs(text_embeds, last_hidden_state, hidden_states)

.. code:: ipython3

    class ConvTransformerWrapper(torch.nn.Module):
        def __init__(self, transformer, config):
            super().__init__()
            self.config = config
            self.transformer = transformer
    
        def forward(self, latents=None, micro_conds=None, pooled_text_emb=None, encoder_hidden_states=None, **kwargs):
            outputs = self.transformer(
                {
                    "latents": latents,
                    "micro_conds": micro_conds,
                    "pooled_text_emb": pooled_text_emb,
                    "encoder_hidden_states": encoder_hidden_states,
                },
                share_inputs=False,
            )
    
            return torch.from_numpy(outputs[0])

.. code:: ipython3

    class ConvVQVAEWrapper(torch.nn.Module):
        def __init__(self, vqvae, dtype, config):
            super().__init__()
            self.vqvae = vqvae
            self.dtype = dtype
            self.config = config
    
        def decode(self, latents=None, force_not_quantize=True, shape=None):
            inputs = {
                "latents": latents,
                "force_not_quantize": force_not_quantize,
                "shape": torch.tensor(shape),
            }
    
            outs = self.vqvae(inputs)
            outs = namedtuple("VQVAE", "sample")(torch.from_numpy(outs[0]))
    
            return outs

And insert wrappers instances in the pipeline:

.. code:: ipython3

    prompt = "kind smiling ghost"
    
    transformer = pipe.transformer
    vqvae = pipe.vqvae
    text_encoder = pipe.text_encoder
    
    pipe.__dict__["_internal_dict"]["_execution_device"] = pipe._execution_device  # this is to avoid some problem that can occur in the pipeline
    pipe.register_modules(
        text_encoder=ConvTextEncoderWrapper(ov_text_encoder, text_encoder.config),
        transformer=ConvTransformerWrapper(ov_transformer, transformer.config),
        vqvae=ConvVQVAEWrapper(ov_vqvae, vqvae.dtype, vqvae.config),
    )
    
    image = pipe(prompt, generator=torch.Generator("cpu").manual_seed(8)).images[0]
    image.save("text2image_256.png")


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140: FutureWarning: Accessing config attribute `_execution_device` directly via 'AmusedPipeline' object attribute is deprecated. Please access '_execution_device' over 'AmusedPipeline's config object instead, e.g. 'scheduler.config._execution_device'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)



.. parsed-literal::

      0%|          | 0/12 [00:00<?, ?it/s]


.. code:: ipython3

    image




.. image:: amused-lightweight-text-to-image-with-output_files/amused-lightweight-text-to-image-with-output_29_0.png



Quantization
------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

According to ``Amused`` pipeline structure, the vision transformer model
takes up significant portion of the overall pipeline execution time. Now
we will show you how to optimize the UNet part using
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to reduce
computation cost and speed up the pipeline. Quantizing the rest of the
pipeline does not significantly improve inference performance but can
lead to a substantial degradation of generations quality.

We also estimate the quality of generations produced by optimized
pipeline with `Inception
Score <https://en.wikipedia.org/wiki/Inception_score>`__ which is often
used to measure quality of text-to-image generation systems.

The steps are the following:

1. Create a calibration dataset for quantization.
2. Run ``nncf.quantize()`` on the model.
3. Save the quantized model using ``openvino.save_model()`` function.
4. Compare inference time and Inception score for original and quantized
   pipelines.

Please select below whether you would like to run quantization to
improve model inference speed.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

.. code:: ipython3

    from notebook_utils import quantization_widget
    
    QUANTIZED_TRANSFORMER_OV_PATH = Path(str(TRANSFORMER_OV_PATH).replace(".xml", "_quantized.xml"))
    
    skip_for_device = "GPU" in device.value
    to_quantize = quantization_widget(not skip_for_device)
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



.. code:: ipython3

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    
    %load_ext skip_kernel_extension

Prepare calibration dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~



We use a portion of
`conceptual_captions <https://huggingface.co/datasets/google-research-datasets/conceptual_captions>`__
dataset from Hugging Face as calibration data. To collect intermediate
model inputs for calibration we customize ``CompiledModel``.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import datasets
    from tqdm.auto import tqdm
    from typing import Any, Dict, List
    import pickle
    import numpy as np
    
    
    def disable_progress_bar(pipeline, disable=True):
        if not hasattr(pipeline, "_progress_bar_config"):
            pipeline._progress_bar_config = {'disable': disable}
        else:
            pipeline._progress_bar_config['disable'] = disable
    
    
    class CompiledModelDecorator(ov.CompiledModel):
        def __init__(self, compiled_model: ov.CompiledModel, data_cache: List[Any] = None, keep_prob: float = 0.5):
            super().__init__(compiled_model)
            self.data_cache = data_cache if data_cache is not None else []
            self.keep_prob = keep_prob
    
        def __call__(self, *args, **kwargs):
            if np.random.rand() <= self.keep_prob:
                self.data_cache.append(*args)
            return super().__call__(*args, **kwargs)
    
    
    def collect_calibration_data(ov_transformer_model, calibration_dataset_size: int) -> List[Dict]:
        calibration_dataset_filepath = Path(f"calibration_data/{calibration_dataset_size}.pkl")
        if not calibration_dataset_filepath.exists():
            calibration_data = []
            pipe.transformer.transformer = CompiledModelDecorator(ov_transformer_model, calibration_data, keep_prob=1.0)
            disable_progress_bar(pipe)
        
            dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", trust_remote_code=True).shuffle(seed=42)
        
            # Run inference for data collection
            pbar = tqdm(total=calibration_dataset_size)
            for batch in dataset:
                prompt = batch["caption"]
                if len(prompt) > pipe.tokenizer.model_max_length:
                    continue
                pipe(prompt, generator=torch.Generator('cpu').manual_seed(0))
                pbar.update(len(calibration_data) - pbar.n)
                if pbar.n >= calibration_dataset_size:
                    break
        
            pipe.transformer.transformer = ov_transformer_model
            disable_progress_bar(pipe, disable=False)
            
            calibration_dataset_filepath.parent.mkdir(exist_ok=True, parents=True)
            with open(calibration_dataset_filepath, 'wb') as f:
                pickle.dump(calibration_data, f)
                
        with open(calibration_dataset_filepath, 'rb') as f:
            calibration_data = pickle.load(f)
        return calibration_data

Run model quantization
~~~~~~~~~~~~~~~~~~~~~~



Run calibration data collection and quantize the vision transformer
model.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
    from nncf.quantization.range_estimator import RangeEstimatorParameters, StatisticsCollectorParameters, StatisticsType, \
        AggregatorType
    import nncf
    
    CALIBRATION_DATASET_SIZE = 12 * 25
    
    if not QUANTIZED_TRANSFORMER_OV_PATH.exists():
        calibration_data = collect_calibration_data(ov_transformer, CALIBRATION_DATASET_SIZE)
        quantized_model = nncf.quantize(
            core.read_model(TRANSFORMER_OV_PATH),
            nncf.Dataset(calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            subset_size=len(calibration_data),
            # We ignore convolutions to improve quality of generations without significant drop in inference speed
            ignored_scope=nncf.IgnoredScope(types=["Convolution"]),
            # Value of 0.85 was obtained using grid search based on Inception Score computed below
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=0.85),
                # During activation statistics collection we ignore 1% of outliers which improves quantization quality
                activations_range_estimator_params=RangeEstimatorParameters(
                    min=StatisticsCollectorParameters(statistics_type=StatisticsType.MIN,
                                                      aggregator_type=AggregatorType.MEAN_NO_OUTLIERS,
                                                      quantile_outlier_prob=0.01),
                    max=StatisticsCollectorParameters(statistics_type=StatisticsType.MAX,
                                                      aggregator_type=AggregatorType.MEAN_NO_OUTLIERS,
                                                      quantile_outlier_prob=0.01)
                )
            )
        )
        ov.save_model(quantized_model, QUANTIZED_TRANSFORMER_OV_PATH)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino



.. parsed-literal::

      0%|          | 0/300 [00:00<?, ?it/s]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140: FutureWarning: Accessing config attribute `_execution_device` directly via 'AmusedPipeline' object attribute is deprecated. Please access '_execution_device' over 'AmusedPipeline's config object instead, e.g. 'scheduler.config._execution_device'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)



.. parsed-literal::

    Output()










.. parsed-literal::

    Output()









.. parsed-literal::

    INFO:nncf:3 ignored nodes were found by types in the NNCFGraph
    INFO:nncf:Not adding activation input quantizer for operation: 53 __module.transformer.embed.conv/aten::_convolution/Convolution
    INFO:nncf:Not adding activation input quantizer for operation: 1986 __module.transformer.mlm_layer.conv1/aten::_convolution/Convolution
    INFO:nncf:Not adding activation input quantizer for operation: 2927 __module.transformer.mlm_layer.conv2/aten::_convolution/Convolution



.. parsed-literal::

    Output()









.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/tensor/tensor.py:100: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/tensor/tensor.py:100: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/tensor/tensor.py:100: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/tensor/tensor.py:100: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/tensor/tensor.py:100: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/tensor/tensor.py:100: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))


Demo generation with quantized pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%skip not $to_quantize.value
    
    original_ov_transformer_model = pipe.transformer.transformer
    pipe.transformer.transformer = core.compile_model(QUANTIZED_TRANSFORMER_OV_PATH, device.value)
    
    image = pipe(prompt, generator=torch.Generator('cpu').manual_seed(8)).images[0]
    image.save('text2image_256_quantized.png')
    
    pipe.transformer.transformer = original_ov_transformer_model
    
    display(image)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140: FutureWarning: Accessing config attribute `_execution_device` directly via 'AmusedPipeline' object attribute is deprecated. Please access '_execution_device' over 'AmusedPipeline's config object instead, e.g. 'scheduler.config._execution_device'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)



.. parsed-literal::

      0%|          | 0/12 [00:00<?, ?it/s]



.. image:: amused-lightweight-text-to-image-with-output_files/amused-lightweight-text-to-image-with-output_38_2.png


Compute Inception Scores and inference time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Below we compute `Inception
Score <https://en.wikipedia.org/wiki/Inception_score>`__ of original and
quantized pipelines on a small subset of images. Images are generated
from prompts of ``conceptual_captions`` validation set. We also measure
the time it took to generate the images for comparison reasons.

Please note that the validation dataset size is small and serves only as
a rough estimate of generation quality.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from torchmetrics.image.inception import InceptionScore
    from torchvision import transforms as transforms
    from itertools import islice
    import time
    
    VALIDATION_DATASET_SIZE = 100
    
    def compute_inception_score(ov_transformer_model_path, validation_set_size, batch_size=100):
        original_ov_transformer_model = pipe.transformer.transformer
        pipe.transformer.transformer = core.compile_model(ov_transformer_model_path, device.value)
        
        disable_progress_bar(pipe)
        dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", "unlabeled", split="validation", trust_remote_code=True).shuffle(seed=42)
        dataset = islice(dataset, validation_set_size)
        
        inception_score = InceptionScore(normalize=True, splits=1)
        
        images = []
        infer_times = []
        for batch in tqdm(dataset, total=validation_set_size, desc="Computing Inception Score"):
            prompt = batch["caption"]
            if len(prompt) > pipe.tokenizer.model_max_length:
                continue
            start_time = time.perf_counter()
            image = pipe(prompt, generator=torch.Generator('cpu').manual_seed(0)).images[0]
            infer_times.append(time.perf_counter() - start_time)
            image = transforms.ToTensor()(image)
            images.append(image)
        
        mean_perf_time = sum(infer_times) / len(infer_times)
            
        while len(images) > 0:
            images_batch = torch.stack(images[-batch_size:])
            images = images[:-batch_size]
            inception_score.update(images_batch)
        kl_mean, kl_std = inception_score.compute()
        
        pipe.transformer.transformer = original_ov_transformer_model
        disable_progress_bar(pipe, disable=False)
        
        return kl_mean, mean_perf_time
    
    
    original_inception_score, original_time = compute_inception_score(TRANSFORMER_OV_PATH, VALIDATION_DATASET_SIZE)
    print(f"Original pipeline Inception Score: {original_inception_score}")
    quantized_inception_score, quantized_time = compute_inception_score(QUANTIZED_TRANSFORMER_OV_PATH, VALIDATION_DATASET_SIZE)
    print(f"Quantized pipeline Inception Score: {quantized_inception_score}")
    print(f"Quantization speed-up: {original_time / quantized_time:.2f}x")


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: Metric `InceptionScore` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
      warnings.warn(\*args, \*\*kwargs)  # noqa: B028



.. parsed-literal::

    Computing Inception Score:   0%|          | 0/100 [00:00<?, ?it/s]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchmetrics/image/inception.py:175: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at ../aten/src/ATen/native/ReduceOps.cpp:1808.)
      return kl.mean(), kl.std()


.. parsed-literal::

    Original pipeline Inception Score: 11.875359535217285



.. parsed-literal::

    Computing Inception Score:   0%|          | 0/100 [00:00<?, ?it/s]


.. parsed-literal::

    Quantized pipeline Inception Score: 11.0730562210083
    Quantization speed-up: 2.09x


Interactive inference
---------------------



Below you can select which pipeline to run: original or quantized.

.. code:: ipython3

    import ipywidgets as widgets
    
    quantized_model_present = QUANTIZED_TRANSFORMER_OV_PATH.exists()
    
    use_quantized_model = widgets.Checkbox(
        value=True if quantized_model_present else False,
        description="Use quantized pipeline",
        disabled=not quantized_model_present,
    )
    
    use_quantized_model




.. parsed-literal::

    Checkbox(value=True, description='Use quantized pipeline')



.. code:: ipython3

    from pathlib import Path
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/amused-lightweight-text-to-image/gradio_helper.py"
        )
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    pipe.transformer.transformer = core.compile_model(
        QUANTIZED_TRANSFORMER_OV_PATH if use_quantized_model.value else TRANSFORMER_OV_PATH,
        device.value,
    )
    
    demo = make_demo(pipe)
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.







