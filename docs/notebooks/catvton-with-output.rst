Virtual Try-On with CatVTON and OpenVINO
========================================

Virtual try-on methods based on diffusion models achieve realistic
try-on effects but replicate the backbone network as a ReferenceNet or
leverage additional image encoders to process condition inputs,
resulting in high training and inference costs. `In this
work <http://arxiv.org/abs/2407.15886>`__, authors rethink the necessity
of ReferenceNet and image encoders and innovate the interaction between
garment and person, proposing CatVTON, a simple and efficient virtual
try-on diffusion model. It facilitates the seamless transfer of in-shop
or worn garments of arbitrary categories to target persons by simply
concatenating them in spatial dimensions as inputs. The efficiency of
the model is demonstrated in three aspects: 1. Lightweight network. Only
the original diffusion modules are used, without additional network
modules. The text encoder and cross attentions for text injection in the
backbone are removed, further reducing the parameters by 167.02M. 2.
Parameter-efficient training. We identified the try-on relevant modules
through experiments and achieved high-quality try-on effects by training
only 49.57M parameters (∼5.51% of the backbone network’s parameters). 3.
Simplified inference. CatVTON eliminates all unnecessary conditions and
preprocessing steps, including pose estimation, human parsing, and text
input, requiring only garment reference, target person image, and mask
for the virtual try-on process. Extensive experiments demonstrate that
CatVTON achieves superior qualitative and quantitative results with
fewer prerequisites and trainable parameters than baseline methods.
Furthermore, CatVTON shows good generalization in in-the-wild scenarios
despite using open-source datasets with only 73K samples.

Teaser image from `CatVTON
GitHub <https://github.com/Zheng-Chong/CatVTON>`__ |teaser|

In this tutorial we consider how to convert and run this model using
OpenVINO. An additional part demonstrates how to run optimization with
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to speed up
pipeline.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Compiling models <#compiling-models>`__
-  `Optimize model using NNCF Post-Training Quantization
   API <#optimize-model-using-nncf-post-training-quantization-api>`__

   -  `Run Post-Training
      Quantization <#run-post-training-quantization>`__
   -  `Run Weights Compression <#run-weights-compression>`__
   -  `Compare model file sizes <#compare-model-file-sizes>`__

-  `Interactive demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |teaser| image:: https://github.com/Zheng-Chong/CatVTON/blob/edited/resource/img/teaser.jpg?raw=true

Prerequisites
-------------



.. code:: ipython3

    import platform
    
    
    if platform.system() == "Darwin":
        %pip install -q "numpy<2.0.0"
    %pip install -q "openvino>=2024.4" "nncf>=2.13.0"
    %pip install -q "torch>=2.1" "diffusers>=0.29.1" torchvision opencv_python --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q fvcore "pillow" "tqdm" "gradio>=4.36" "omegaconf==2.4.0.dev3" av pycocotools cloudpickle scipy accelerate "transformers>=4.27.3"

.. code:: ipython3

    import requests
    
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py",
    )
    open("cmd_helper.py", "w").write(r.text)

.. code:: ipython3

    from cmd_helper import clone_repo
    
    
    clone_repo("https://github.com/Zheng-Chong/CatVTON.git", "3b795364a4d2f3b5adb365f39cdea376d20bc53c")

Convert the model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.

``ov_catvton_helper.py`` script contains helper function for models
downloading and models conversion, please check its content if you
interested in conversion details.

To download checkpoints and load models, just call the helper function
``download_models``. It takes care about it. Functions
``convert_pipeline_models`` and ``convert_automasker_models`` will
convert models from pipeline and ``automasker`` in OpenVINO format.

The original pipeline contains VAE encoder and decoder and UNET.
|CatVTON-overview|

The ``automasker`` contains ``DensePose`` with
``detectron2.GeneralizedRCNN`` model and ``SCHP`` (``LIP`` and ``ATR``
version).

.. |CatVTON-overview| image:: https://github.com/user-attachments/assets/e35c8dab-1c54-47b1-a73b-2a62e6cdca7c

.. code:: ipython3

    from ov_catvton_helper import download_models, convert_pipeline_models, convert_automasker_models
    
    pipeline, mask_processor, automasker = download_models()
    convert_pipeline_models(pipeline)
    convert_automasker_models(automasker)

Compiling models
----------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import openvino as ov
    
    from notebook_utils import device_widget
    
    
    core = ov.Core()
    
    device = device_widget()
    
    device

``get_compiled_pipeline`` and ``get_compiled_automasker`` functions
defined in ``ov_catvton_helper.py`` provides convenient way for getting
the pipeline and the ``automasker`` with compiled ov-models that are
compatible with the original interface. It accepts the original pipeline
and ``automasker``, inference device and directories with converted
models as arguments. Under the hood we create callable wrapper classes
for compiled models to allow interaction with original pipelines. Note
that all of wrapper classes return ``torch.Tensor``\ s instead of
``np.array``\ s. And then insert wrappers instances in the pipeline.

.. code:: ipython3

    from ov_catvton_helper import (
        get_compiled_pipeline,
        get_compiled_automasker,
        VAE_ENCODER_PATH,
        VAE_DECODER_PATH,
        UNET_PATH,
        DENSEPOSE_PROCESSOR_PATH,
        SCHP_PROCESSOR_ATR,
        SCHP_PROCESSOR_LIP,
    )
    
    pipeline = get_compiled_pipeline(pipeline, core, device, VAE_ENCODER_PATH, VAE_DECODER_PATH, UNET_PATH)
    automasker = get_compiled_automasker(automasker, core, device, DENSEPOSE_PROCESSOR_PATH, SCHP_PROCESSOR_ATR, SCHP_PROCESSOR_LIP)

Optimize model using NNCF Post-Training Quantization API
--------------------------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) for the UNet
model, and 4-bit weight compression for the remaining models.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time. You can disable
   it using widget below:

.. code:: ipython3

    from notebook_utils import quantization_widget
    
    to_quantize = quantization_widget()
    
    to_quantize

Let’s load ``skip magic`` extension to skip quantization if
``to_quantize`` is not selected

.. code:: ipython3

    optimized_pipe = None
    optimized_automasker = None
    
    # Fetch skip_kernel_extension module
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    
    %load_ext skip_kernel_extension

Run Post-Training Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The optimization process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` for getting an optimized model.
3. Serialize an OpenVINO IR model, using the ``openvino.save_model``
   function.

We use a couple of images from the original repository as calibration
data.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from pathlib import Path
    from catvton_quantization_helper import collect_calibration_data, UNET_INT8_PATH
    
    dataset = [
        (
            Path("CatVTON/resource/demo/example/person/men/model_5.png"),
            Path("CatVTON/resource/demo/example/condition/upper/24083449_54173465_2048.jpg"),
        ),
        (
            Path("CatVTON/resource/demo/example/person/women/2-model_4.png"),
            Path("CatVTON/resource/demo/example/condition/overall/21744571_51588794_1000.jpg"),
        ),
    ]
    
    if not UNET_INT8_PATH.exists():
        subset_size = 100
        calibration_data = collect_calibration_data(pipeline, automasker, mask_processor, dataset, subset_size)

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import nncf
    from ov_catvton_helper import UNET_PATH
    
    if not UNET_INT8_PATH.exists():
        unet = core.read_model(UNET_PATH)
        quantized_model = nncf.quantize(
            model=unet,
            calibration_dataset=nncf.Dataset(calibration_data),
            subset_size=subset_size,
            model_type=nncf.ModelType.TRANSFORMER,
        )
        ov.save_model(quantized_model, UNET_INT8_PATH)

Run Weights Compression
~~~~~~~~~~~~~~~~~~~~~~~



Quantizing of the remaining components of the pipeline does not
significantly improve inference performance but can lead to a
substantial degradation of accuracy. The weight compression will be
applied to footprint reduction.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from catvton_quantization_helper import compress_models
    
    compress_models(core)

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from catvton_quantization_helper import (
        VAE_ENCODER_INT4_PATH,
        VAE_DECODER_INT4_PATH,
        DENSEPOSE_PROCESSOR_INT4_PATH,
        SCHP_PROCESSOR_ATR_INT4,
        SCHP_PROCESSOR_LIP_INT4,
    )
    
    optimized_pipe, _, optimized_automasker = download_models()
    optimized_pipe = get_compiled_pipeline(optimized_pipe, core, device, VAE_ENCODER_INT4_PATH, VAE_DECODER_INT4_PATH, UNET_INT8_PATH)
    optimized_automasker = get_compiled_automasker(optimized_automasker, core, device, DENSEPOSE_PROCESSOR_INT4_PATH, SCHP_PROCESSOR_ATR_INT4, SCHP_PROCESSOR_LIP_INT4)

Compare model file sizes
~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    from catvton_quantization_helper import compare_models_size
    
    compare_models_size()


.. parsed-literal::

    vae_encoder compression rate: 2.011
    vae_decoder compression rate: 2.007
    unet compression rate: 1.995
    densepose_processor compression rate: 2.019
    schp_processor_atr compression rate: 1.993
    schp_processor_lip compression rate: 1.993
    

Interactive inference
---------------------



Please select below whether you would like to use the quantized models
to launch the interactive demo.

.. code:: ipython3

    from ov_catvton_helper import get_pipeline_selection_option
    
    use_quantized_models = get_pipeline_selection_option(optimized_pipe)
    
    use_quantized_models

.. code:: ipython3

    from gradio_helper import make_demo
    
    pipe = optimized_pipe if use_quantized_models.value else pipeline
    masker = optimized_automasker if use_quantized_models.value else automasker
    
    output_dir = "output"
    demo = make_demo(pipe, mask_processor, masker, output_dir)
    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(debug=True, share=True)
