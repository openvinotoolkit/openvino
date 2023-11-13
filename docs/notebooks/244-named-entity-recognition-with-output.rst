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
---
- `Prerequisites <#prerequisites>`__

-  `Download the NER model <#download-the-ner-model>`__
-  `Quantize the model, using Hugging Face Optimum
   API <#quantize-the-model-using-hugging-face-optimum-api>`__
-  `Prepare demo for Named Entity Recognition OpenVINO
   Runtime <#prepare-demo-for-named-entity-recognition-openvino-runtime>`__
-  `Compare the Original and Quantized
   Models <#compare-the-original-and-quantized-models>`__

   -  `Compare performance <#compare-performance>`__
   -  `Compare size of the
      models <#compare-size-of-the-models>`__

Prerequisites 
-------------------------------------------------------

.. code:: ipython3

    %pip install -q "diffusers>=0.17.1" "openvino>=2023.1.0" "nncf>=2.5.0" "gradio" "onnx>=1.11.0" "onnxruntime>=1.14.0" "transformers>=4.31.0"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"

Download the NER model 
----------------------------------------------------------------

We load the
`distilbert-base-cased-finetuned-conll03-english <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__
model from the `Hugging Face Hub <https://huggingface.co/models>`__ with
`Hugging Face Transformers
library <https://huggingface.co/docs/transformers/index>`__.

Model class initialization starts with calling ``from_pretrained``
method. To easily save the model, you can use the ``save_pretrained()``
method.

.. code:: ipython3

    from transformers import AutoTokenizer, AutoModelForTokenClassification
    
    model_id = "elastic/distilbert-base-cased-finetuned-conll03-english"
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    
    original_ner_model_dir = 'original_ner_model'
    model.save_pretrained(original_ner_model_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)


.. parsed-literal::

    2023-09-19 19:03:57.913343: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-09-19 19:03:57.950536: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-09-19 19:03:58.511125: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Quantize the model, using Hugging Face Optimum API 
--------------------------------------------------------------------------------------------

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
    from optimum.intel import OVQuantizer
    
    from optimum.intel import OVModelForTokenClassification
    
    def preprocess_fn(data, tokenizer):
        examples = []
        for data_chunk in data["tokens"]:
            examples.append(' '.join(data_chunk))
    
        return tokenizer(
            examples, padding=True, truncation=True, max_length=128
        )
    
    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        "conll2003",
        preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
        num_samples=100,
        dataset_split="train",
        preprocess_batch=True,
    )
    
    # The directory where the quantized model will be saved
    quantized_ner_model_dir = "quantized_ner_model"
    
    # Apply static quantization and save the resulting model in the OpenVINO IR format
    quantizer.quantize(calibration_dataset=calibration_dataset, save_directory=quantized_ner_model_dir)
    
    # Load the quantized model
    optimized_model = OVModelForTokenClassification.from_pretrained(quantized_ner_model_dir)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    /home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
      warnings.warn(
    Found cached dataset conll2003 (/home/ea/.cache/huggingface/datasets/conll2003/conll2003/1.0.0/9a4d16a94f8674ba3466315300359b0acd891b68b6c8743ddf60b9c702adce98)
    Loading cached shuffled indices for dataset at /home/ea/.cache/huggingface/datasets/conll2003/conll2003/1.0.0/9a4d16a94f8674ba3466315300359b0acd891b68b6c8743ddf60b9c702adce98/cache-2fe5320fac60946d.arrow



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?ba/s]


.. parsed-literal::

    No configuration describing the quantization process was provided, a default OVConfig will be generated.


.. parsed-literal::

    INFO:nncf:Not adding activation input quantizer for operation: 3 DistilBertForTokenClassification/DistilBertModel[distilbert]/Embeddings[embeddings]/NNCFEmbedding[position_embeddings]/embedding_0
    INFO:nncf:Not adding activation input quantizer for operation: 2 DistilBertForTokenClassification/DistilBertModel[distilbert]/Embeddings[embeddings]/NNCFEmbedding[word_embeddings]/embedding_0
    INFO:nncf:Not adding activation input quantizer for operation: 4 DistilBertForTokenClassification/DistilBertModel[distilbert]/Embeddings[embeddings]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 5 DistilBertForTokenClassification/DistilBertModel[distilbert]/Embeddings[embeddings]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 6 DistilBertForTokenClassification/DistilBertModel[distilbert]/Embeddings[embeddings]/Dropout[dropout]/dropout_0
    INFO:nncf:Not adding activation input quantizer for operation: 16 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[0]/MultiHeadSelfAttention[attention]/__truediv___0
    INFO:nncf:Not adding activation input quantizer for operation: 25 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[0]/MultiHeadSelfAttention[attention]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 30 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[0]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 31 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[0]/NNCFLayerNorm[sa_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 35 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[0]/__add___1
    INFO:nncf:Not adding activation input quantizer for operation: 36 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[0]/NNCFLayerNorm[output_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 46 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[1]/MultiHeadSelfAttention[attention]/__truediv___0
    INFO:nncf:Not adding activation input quantizer for operation: 55 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[1]/MultiHeadSelfAttention[attention]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 60 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[1]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 61 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[1]/NNCFLayerNorm[sa_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 65 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[1]/__add___1
    INFO:nncf:Not adding activation input quantizer for operation: 66 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[1]/NNCFLayerNorm[output_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 76 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[2]/MultiHeadSelfAttention[attention]/__truediv___0
    INFO:nncf:Not adding activation input quantizer for operation: 85 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[2]/MultiHeadSelfAttention[attention]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 90 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[2]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 91 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[2]/NNCFLayerNorm[sa_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 95 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[2]/__add___1
    INFO:nncf:Not adding activation input quantizer for operation: 96 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[2]/NNCFLayerNorm[output_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 106 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[3]/MultiHeadSelfAttention[attention]/__truediv___0
    INFO:nncf:Not adding activation input quantizer for operation: 115 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[3]/MultiHeadSelfAttention[attention]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 120 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[3]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 121 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[3]/NNCFLayerNorm[sa_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 125 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[3]/__add___1
    INFO:nncf:Not adding activation input quantizer for operation: 126 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[3]/NNCFLayerNorm[output_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 136 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[4]/MultiHeadSelfAttention[attention]/__truediv___0
    INFO:nncf:Not adding activation input quantizer for operation: 145 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[4]/MultiHeadSelfAttention[attention]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 150 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[4]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 151 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[4]/NNCFLayerNorm[sa_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 155 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[4]/__add___1
    INFO:nncf:Not adding activation input quantizer for operation: 156 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[4]/NNCFLayerNorm[output_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 166 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[5]/MultiHeadSelfAttention[attention]/__truediv___0
    INFO:nncf:Not adding activation input quantizer for operation: 175 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[5]/MultiHeadSelfAttention[attention]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 180 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[5]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 181 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[5]/NNCFLayerNorm[sa_layer_norm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 185 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[5]/__add___1
    INFO:nncf:Not adding activation input quantizer for operation: 186 DistilBertForTokenClassification/DistilBertModel[distilbert]/Transformer[transformer]/ModuleList[layer]/TransformerBlock[5]/NNCFLayerNorm[output_layer_norm]/layer_norm_0
    INFO:nncf:Collecting tensor statistics |█               | 33 / 300
    INFO:nncf:Collecting tensor statistics |███             | 66 / 300
    INFO:nncf:Collecting tensor statistics |█████           | 99 / 300
    INFO:nncf:Compiling and loading torch extension: quantized_functions_cpu...
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    INFO:nncf:Finished loading torch extension: quantized_functions_cpu


.. parsed-literal::

    Using framework PyTorch: 2.0.1+cpu


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /home/ea/work/ov_venv/lib/python3.8/site-packages/nncf/torch/dynamic_graph/wrappers.py:81: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      result = operator(\*args, \*\*kwargs)
    Configuration saved in quantized_ner_model/openvino_config.json
    Compiling the model...
    Set CACHE_DIR to quantized_ner_model/model_cache


Prepare demo for Named Entity Recognition OpenVINO Runtime 
----------------------------------------------------------------------------------------------------

As the Optimum Inference models are API compatible with Hugging Face
Transformers models, we can just use ``pipleine()`` from `Hugging Face
Transformers API <https://huggingface.co/docs/transformers/index>`__ for
inference.

.. code:: ipython3

    from transformers import pipeline
    
    ner_pipeline_optimized = pipeline("token-classification", model=optimized_model, tokenizer=tokenizer)

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
    
    demo = gr.Interface(run_ner,
                        gr.Textbox(placeholder="Enter sentence here...", label="Input Text"), 
                        gr.HighlightedText(label="Output Text"),
                        examples=examples,
                        allow_flagging="never")
    
    if __name__ == "__main__":
        try:
            demo.launch(debug=False)
        except Exception:
            demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



.. .. raw:: html

..    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>


.. parsed-literal::

    Keyboard interruption in main thread... closing server.


Compare the Original and Quantized Models 
-----------------------------------------------------------------------------------

Compare the original
`distilbert-base-cased-finetuned-conll03-english <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__
model with quantized and converted to OpenVINO IR format models to see
the difference.

Compare performance 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ner_pipeline_original = pipeline("token-classification", model=model, tokenizer=tokenizer)

.. code:: ipython3

    import time
    import numpy as np
    
    def calc_perf(ner_pipeline):
        inference_times = []
    
        for data in calibration_dataset:
            text = ' '.join(data['tokens'])
            start = time.perf_counter()
            ner_pipeline(text)
            end = time.perf_counter()
            inference_times.append(end - start)
    
        return np.median(inference_times)
    
    
    print(
        f"Median inference time of quantized model: {calc_perf(ner_pipeline_optimized)} "
    )
    
    print(
        f"Median inference time of original model: {calc_perf(ner_pipeline_original)} "
    )


.. parsed-literal::

    Median inference time of quantized model: 0.008145123501890339 
    Median inference time of original model: 0.09339697850373341 


Compare size of the models 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from pathlib import Path
    
    print(f'Size of original model in Bytes is {Path(original_ner_model_dir, "pytorch_model.bin").stat().st_size}')
    print(f'Size of quantized model in Bytes is {Path(quantized_ner_model_dir, "openvino_model.bin").stat().st_size}')


.. parsed-literal::

    Size of original model in Bytes is 260824741
    Size of quantized model in Bytes is 133539000

