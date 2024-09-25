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

    %pip install -q "diffusers>=0.17.1" "openvino>=2023.1.0" "nncf>=2.5.0" "gradio>=4.19" "onnx>=1.11.0,<1.16.2" "transformers>=4.33.0" "torch>=2.1" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"

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

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import device_widget
    
    device = device_widget()
    
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
    
    
    def run_ner(text):
        output = ner_pipeline_optimized(text)
        return {"text": text, "entities": output}
    
    
    demo = gr.Interface(
        fn=run_ner,
        inputs=gr.Textbox(placeholder="Enter sentence here...", label="Input Text"),
        outputs=gr.HighlightedText(label="Output Text"),
        examples=[
            "My name is Wolfgang and I live in Berlin.",
        ],
        allow_flagging="never",
    )
    
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/

.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
