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

.. _top:

**Table of contents**:

- `Prerequisites <#prerequisites>`__
- `Download the NER model <#download-the-ner-model>`__
- `Quantize the model, using Hugging Face Optimum API <#quantize-the-model-using-hugging-face-optimum-api>`__
- `Prepare demo for Named Entity Recognition OpenVINO Runtime <#prepare-demo-for-named-entity-recognition-openvino-runtime>`__
- `Compare the Original and Quantized Models <#compare-the-original-and-quantized-models>`__

  - `Compare performance <#compare-performance>`__
  - `Compare size of the models <#compare-size-of-the-models>`__

Prerequisites `⇑ <#top>`__
###############################################################################################################################


.. code:: ipython3

    !pip install -q "diffusers>=0.17.1" "openvino-dev>=2023.0.0" "nncf>=2.5.0" "gradio" "onnx>=1.11.0" "onnxruntime>=1.14.0" "optimum-intel>=1.9.1" "transformers>=4.31.0"


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    audiocraft 0.0.2a2 requires xformers, which is not installed.
    audiocraft 0.0.2a2 requires torch>=2.0.0, but you have torch 1.13.1+cpu which is incompatible.
    audiocraft 0.0.2a2 requires torchaudio>=2.0.0, but you have torchaudio 0.13.1+cpu which is incompatible.
    deepfloyd-if 1.0.2rc0 requires accelerate~=0.15.0, but you have accelerate 0.22.0.dev0 which is incompatible.
    deepfloyd-if 1.0.2rc0 requires diffusers~=0.16.0, but you have diffusers 0.18.2 which is incompatible.
    deepfloyd-if 1.0.2rc0 requires transformers~=4.25.1, but you have transformers 4.30.2 which is incompatible.
    paddleclas 2.5.1 requires faiss-cpu==1.7.1.post2, but you have faiss-cpu 1.7.4 which is incompatible.
    paddleclas 2.5.1 requires gast==0.3.3, but you have gast 0.4.0 which is incompatible.
    ppgan 2.1.0 requires librosa==0.8.1, but you have librosa 0.9.2 which is incompatible.
    ppgan 2.1.0 requires opencv-python<=4.6.0.66, but you have opencv-python 4.7.0.72 which is incompatible.
    pytorch-lightning 1.6.5 requires protobuf<=3.20.1, but you have protobuf 3.20.3 which is incompatible.
    spacy 3.5.2 requires pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4, but you have pydantic 2.0.3 which is incompatible.
    thinc 8.1.10 requires pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4, but you have pydantic 2.0.3 which is incompatible.
    visualdl 2.5.2 requires gradio==3.11.0, but you have gradio 3.36.1 which is incompatible.
    
    [notice] A new release of pip is available: 23.1.2 -> 23.2
    [notice] To update, run: pip install --upgrade pip


Download the NER model `⇑ <#top>`__
###############################################################################################################################


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

    Downloading (…)lve/main/config.json:   0%|          | 0.00/954 [00:00<?, ?B/s]



.. parsed-literal::

    Downloading model.safetensors:   0%|          | 0.00/261M [00:00<?, ?B/s]



.. parsed-literal::

    Downloading (…)okenizer_config.json:   0%|          | 0.00/257 [00:00<?, ?B/s]



.. parsed-literal::

    Downloading (…)solve/main/vocab.txt:   0%|          | 0.00/213k [00:00<?, ?B/s]



.. parsed-literal::

    Downloading (…)cial_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]


Quantize the model, using Hugging Face Optimum API `⇑ <#top>`__
###############################################################################################################################


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

    2023-07-17 14:40:49.402855: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-07-17 14:40:49.442756: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-07-17 14:40:50.031065: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    comet_ml is installed but `COMET_API_KEY` is not set.



.. parsed-literal::

    Downloading builder script:   0%|          | 0.00/9.57k [00:00<?, ?B/s]



.. parsed-literal::

    Downloading metadata:   0%|          | 0.00/3.73k [00:00<?, ?B/s]



.. parsed-literal::

    Downloading readme:   0%|          | 0.00/12.3k [00:00<?, ?B/s]


.. parsed-literal::

    Downloading and preparing dataset conll2003/conll2003 to /home/ea/.cache/huggingface/datasets/conll2003/conll2003/1.0.0/9a4d16a94f8674ba3466315300359b0acd891b68b6c8743ddf60b9c702adce98...



.. parsed-literal::

    Downloading data:   0%|          | 0.00/983k [00:00<?, ?B/s]



.. parsed-literal::

    Generating train split:   0%|          | 0/14041 [00:00<?, ? examples/s]



.. parsed-literal::

    Generating validation split:   0%|          | 0/3250 [00:00<?, ? examples/s]



.. parsed-literal::

    Generating test split:   0%|          | 0/3453 [00:00<?, ? examples/s]


.. parsed-literal::

    Dataset conll2003 downloaded and prepared to /home/ea/.cache/huggingface/datasets/conll2003/conll2003/1.0.0/9a4d16a94f8674ba3466315300359b0acd891b68b6c8743ddf60b9c702adce98. Subsequent calls will reuse this data.



.. parsed-literal::

    Map:   0%|          | 0/100 [00:00<?, ? examples/s]


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
    INFO:nncf:Collecting tensor statistics |█               | 4 / 38
    INFO:nncf:Collecting tensor statistics |███             | 8 / 38
    INFO:nncf:Collecting tensor statistics |█████           | 12 / 38
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
    WARNING:nncf:You are setting `forward` on an NNCF-processed model object.
    NNCF relies on custom-wrapping the `forward` call in order to function properly.
    Arbitrary adjustments to the forward function on an NNCFNetwork object have undefined behaviour.
    If you need to replace the underlying forward function of the original model so that NNCF should be using that instead of the original forward function that NNCF saved during the compressed model creation, you can do this by calling:
    model.nncf.set_original_unbound_forward(fn)
    if `fn` has an unbound 0-th `self` argument, or
    with model.nncf.temporary_bound_original_forward(fn): ...
    if `fn` already had 0-th `self` argument bound or never had it in the first place.


.. code::

   /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/nncf/torch/nncf_network.py:938: FutureWarning: Old style of accessing NNCF-specific attributes and methods on NNCFNetwork objects is deprecated. Access the NNCF-specific attrs through the NNCFInterface, which is set up as an `nncf` attribute on the compressed model object.
   For instance, instead of `compressed_model.get_graph()` you should now write `compressed_model.nncf.get_graph()`.
   The old style will be removed after NNCF v2.5.0
     warning_deprecated(
   /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/nncf/torch/quantization/layers.py:338: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
     return self._level_low.item()
   /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/nncf/torch/quantization/layers.py:346: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
     return self._level_high.item()
   /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/nncf/torch/dynamic_graph/wrappers.py:81: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
     result = operator(*args, **kwargs)
   /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/nncf/torch/quantization/quantize_functions.py:140: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
     output = g.op(
   /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/torch/onnx/_patch_torch.py:81: UserWarning: The shape inference of org.openvinotoolkit::FakeQuantize type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (Triggered internally at ../torch/csrc/jit/passes/onnx/shape_type_inference.cpp:1884.)
     _C._jit_pass_onnx_node_shape_type_inference(
   /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/torch/onnx/utils.py:687: UserWarning: The shape inference of org.openvinotoolkit::FakeQuantize type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (Triggered internally at ../torch/csrc/jit/passes/onnx/shape_type_inference.cpp:1884.)
     _C._jit_pass_onnx_graph_shape_type_inference(
   /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/torch/onnx/utils.py:1178: UserWarning: The shape inference of org.openvinotoolkit::FakeQuantize type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (Triggered internally at ../torch/csrc/jit/passes/onnx/shape_type_inference.cpp:1884.)
     _C._jit_pass_onnx_graph_shape_type_inference(

.. parsed-literal::

    WARNING:nncf:You are setting `forward` on an NNCF-processed model object.
    NNCF relies on custom-wrapping the `forward` call in order to function properly.
    Arbitrary adjustments to the forward function on an NNCFNetwork object have undefined behaviour.
    If you need to replace the underlying forward function of the original model so that NNCF should be using that instead of the original forward function that NNCF saved during the compressed model creation, you can do this by calling:
    model.nncf.set_original_unbound_forward(fn)
    if `fn` has an unbound 0-th `self` argument, or
    with model.nncf.temporary_bound_original_forward(fn): ...
    if `fn` already had 0-th `self` argument bound or never had it in the first place.


.. parsed-literal::

    Configuration saved in quantized_ner_model/openvino_config.json
    Compiling the model...


Prepare demo for Named Entity Recognition OpenVINO Runtime `⇑ <#top>`__
###############################################################################################################################


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
            demo.launch(debug=True)
        except Exception:
            demo.launch(share=True, debug=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    
    Thanks for being a Gradio user! If you have questions or feedback, please join our Discord server and chat with us: https://discord.gg/feTf9x3ZSB
    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



.. .. raw:: html

..     <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>


.. parsed-literal::

    Keyboard interruption in main thread... closing server.


Compare the Original and Quantized Models `⇑ <#top>`__
###############################################################################################################################


Compare the original
`distilbert-base-cased-finetuned-conll03-english <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__
model with quantized and converted to OpenVINO IR format models to see
the difference.

Compare performance `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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

    Median inference time of quantized model: 0.008888308017048985 


Compare size of the models `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. code:: ipython3

    from pathlib import Path
    
    print(f'Size of original model in Bytes is {Path(original_ner_model_dir, "pytorch_model.bin").stat().st_size}')
    print(f'Size of quantized model in Bytes is {Path(quantized_ner_model_dir, "openvino_model.bin").stat().st_size}')
