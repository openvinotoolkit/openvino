Named entity recognition with OpenVINO™
=======================================

The Named Entity Recognition(NER) is a natural language processing
method that involves the detecting of key information in the
unstructured text and categorizing it into pre-defined categories. These
categories or named entities refer to the key subjects of text, such as
names, locations, companies and etc.

NER is a good method for the situations when a high-level overview of a
large amount of text is needed. NER can be helpful with such task as
analyzing key information in unstructured text or automates the
information extraction of large amounts of data.

This tutorial shows how to perform named entity recognition using
OpenVINO. We will use the pre-trained model
`elastic/distilbert-base-cased-finetuned-conll03-english <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__.
It is DistilBERT based model, trained on
`conll03 english dataset <https://huggingface.co/datasets/conll2003>`__.
The model can recognize four named entities in text: persons, locations,
organizations and names of miscellaneous entities that do not belong to
the previous three groups. The model is sensitive to capital letters.

To simplify the user experience, the `Hugging Face
Optimum <https://huggingface.co/docs/optimum>`__ library is used to
convert the model to OpenVINO™ IR format and quantize it.

**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Download the NER model <#download-the-ner-model>`__
-  `Quantize the model, using Hugging Face Optimum
   API <#quantize-the-model-using-hugging-face-optimum-api>`__
-  `Compare the Original and Quantized
   Models <#compare-the-original-and-quantized-models>`__

   -  `Compare performance <#compare-performance>`__
   -  `Compare size of the models <#compare-size-of-the-models>`__

-  `Prepare demo for Named Entity Recognition OpenVINO
   Runtime <#prepare-demo-for-named-entity-recognition-openvino-runtime>`__

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "diffusers>=0.17.1" "openvino>=2023.1.0" "nncf>=2.5.0" "gradio>=4.19" "onnx>=1.11.0" "transformers>=4.33.0" "torch>=2.1" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Download the NER model
----------------------



We load the
`distilbert-base-cased-finetuned-conll03-english <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__
model from the `Hugging Face Hub <https://huggingface.co/models>`__ with
`Hugging Face Transformers
library <https://huggingface.co/docs/transformers/index>`__\ and Optimum
Intel with OpenVINO integration.

``OVModelForTokenClassification`` is represent model class for Named
Entity Recognition task in Optimum Intel. Model class initialization
starts with calling ``from_pretrained`` method. For conversion original
PyTorch model to OpenVINO format on the fly, ``export=True`` parameter
should be used. To easily save the model, you can use the
``save_pretrained()`` method. After saving the model on disk, we can use
pre-converted model for next usage, and speedup deployment process.

.. code:: ipython3

    from pathlib import Path
    from transformers import AutoTokenizer
    from optimum.intel import OVModelForTokenClassification
    
    original_ner_model_dir = Path("original_ner_model")
    
    model_id = "elastic/distilbert-base-cased-finetuned-conll03-english"
    if not original_ner_model_dir.exists():
        model = OVModelForTokenClassification.from_pretrained(model_id, export=True)
    
        model.save_pretrained(original_ner_model_dir)
    else:
        model = OVModelForTokenClassification.from_pretrained(model_id, export=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    2024-04-05 18:35:04.594311: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-05 18:35:04.596755: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-04-05 18:35:04.628293: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-04-05 18:35:04.628326: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-04-05 18:35:04.628349: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-04-05 18:35:04.634704: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-04-05 18:35:04.635314: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-04-05 18:35:05.607762: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /home/ea/miniconda3/lib/python3.11/site-packages/transformers/utils/import_utils.py:519: FutureWarning: `is_torch_tpu_available` is deprecated and will be removed in 4.41.0. Please use the `is_torch_xla_available` instead.
      warnings.warn(
    Framework not specified. Using pt to export the model.
    Using the export variant default. Available variants are:
        - default: The default ONNX variant.
    Using framework PyTorch: 2.1.2+cpu
    /home/ea/miniconda3/lib/python3.11/site-packages/transformers/modeling_utils.py:4225: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /home/ea/miniconda3/lib/python3.11/site-packages/nncf/torch/dynamic_graph/wrappers.py:80: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      op1 = operator(\*args, \*\*kwargs)
    Compiling the model to CPU ...


Quantize the model, using Hugging Face Optimum API
--------------------------------------------------



Post-training static quantization introduces an additional calibration
step where data is fed through the network in order to compute the
activations quantization parameters. For quantization it will be used
`Hugging Face Optimum Intel
API <https://huggingface.co/docs/optimum/intel/index>`__.

To handle the NNCF quantization process we use class
`OVQuantizer <https://huggingface.co/docs/optimum/intel/reference_ov#optimum.intel.OVQuantizer>`__.
The quantization with Hugging Face Optimum Intel API contains the next
steps: \* Model class initialization starts with calling
``from_pretrained()`` method. \* Next we create calibration dataset with
``get_calibration_dataset()`` to use for the post-training static
quantization calibration step. \* After we quantize a model and save the
resulting model in the OpenVINO IR format to save_directory with
``quantize()`` method. \* Then we load the quantized model. The Optimum
Inference models are API compatible with Hugging Face Transformers
models and we can just replace ``AutoModelForXxx`` class with the
corresponding ``OVModelForXxx`` class. So we use
``OVModelForTokenClassification`` to load the model.

.. code:: ipython3

    from functools import partial
    from optimum.intel import OVQuantizer, OVConfig, OVQuantizationConfig
    
    from optimum.intel import OVModelForTokenClassification
    
    
    def preprocess_fn(data, tokenizer):
        examples = []
        for data_chunk in data["tokens"]:
            examples.append(" ".join(data_chunk))
    
        return tokenizer(examples, padding=True, truncation=True, max_length=128)
    
    
    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        "conll2003",
        preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
        num_samples=100,
        dataset_split="train",
        preprocess_batch=True,
        trust_remote_code=True,
    )
    
    # The directory where the quantized model will be saved
    quantized_ner_model_dir = "quantized_ner_model"
    
    # Apply static quantization and save the resulting model in the OpenVINO IR format
    ov_config = OVConfig(quantization_config=OVQuantizationConfig(num_samples=len(calibration_dataset)))
    quantizer.quantize(
        calibration_dataset=calibration_dataset,
        save_directory=quantized_ner_model_dir,
        ov_config=ov_config,
    )


.. parsed-literal::

    /home/ea/miniconda3/lib/python3.11/site-packages/datasets/load.py:2516: FutureWarning: 'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.
    You can remove this warning by passing 'token=<use_auth_token>' instead.
      warnings.warn(



.. parsed-literal::

    Output()


















.. parsed-literal::

    Output()

















.. parsed-literal::

    INFO:nncf:18 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:25 ignored nodes were found by name in the NNCFGraph



.. parsed-literal::

    Output()


















.. parsed-literal::

    Output()

















.. code:: ipython3

    import ipywidgets as widgets
    import openvino as ov
    
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=3, options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='AUTO')



.. code:: ipython3

    # Load the quantized model
    optimized_model = OVModelForTokenClassification.from_pretrained(quantized_ner_model_dir, device=device.value)


.. parsed-literal::

    Compiling the model to AUTO ...


Compare the Original and Quantized Models
-----------------------------------------



Compare the original
`distilbert-base-cased-finetuned-conll03-english <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__
model with quantized and converted to OpenVINO IR format models to see
the difference.

Compare performance
~~~~~~~~~~~~~~~~~~~



As the Optimum Inference models are API compatible with Hugging Face
Transformers models, we can just use ``pipleine()`` from `Hugging Face
Transformers API <https://huggingface.co/docs/transformers/index>`__ for
inference.

.. code:: ipython3

    from transformers import pipeline
    
    ner_pipeline_optimized = pipeline("token-classification", model=optimized_model, tokenizer=tokenizer)
    
    ner_pipeline_original = pipeline("token-classification", model=model, tokenizer=tokenizer)

.. code:: ipython3

    import time
    import numpy as np
    
    
    def calc_perf(ner_pipeline):
        inference_times = []
    
        for data in calibration_dataset:
            text = " ".join(data["tokens"])
            start = time.perf_counter()
            ner_pipeline(text)
            end = time.perf_counter()
            inference_times.append(end - start)
    
        return np.median(inference_times)
    
    
    print(f"Median inference time of quantized model: {calc_perf(ner_pipeline_optimized)} ")
    
    print(f"Median inference time of original model: {calc_perf(ner_pipeline_original)} ")


.. parsed-literal::

    Median inference time of quantized model: 0.0063508255407214165 
    Median inference time of original model: 0.007429798366501927 


Compare size of the models
~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from pathlib import Path
    
    fp_model_file = Path(original_ner_model_dir) / "openvino_model.bin"
    print(f"Size of original model in Bytes is {fp_model_file.stat().st_size}")
    print(f'Size of quantized model in Bytes is {Path(quantized_ner_model_dir, "openvino_model.bin").stat().st_size}')


.. parsed-literal::

    Size of original model in Bytes is 260795516
    Size of quantized model in Bytes is 65802712


Prepare demo for Named Entity Recognition OpenVINO Runtime
----------------------------------------------------------



Now, you can try NER model on own text. Put your sentence to input text
box, click Submit button, the model label the recognized entities in the
text.

.. code:: ipython3

    import gradio as gr
    
    examples = [
        "My name is Wolfgang and I live in Berlin.",
    ]
    
    
    def run_ner(text):
        output = ner_pipeline_optimized(text)
        return {"text": text, "entities": output}
    
    
    demo = gr.Interface(
        run_ner,
        gr.Textbox(placeholder="Enter sentence here...", label="Input Text"),
        gr.HighlightedText(label="Output Text"),
        examples=examples,
        allow_flagging="never",
    )
    
    if __name__ == "__main__":
        try:
            demo.launch(debug=False)
        except Exception:
            demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
