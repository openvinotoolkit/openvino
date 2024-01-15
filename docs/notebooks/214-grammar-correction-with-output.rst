Grammatical Error Correction with OpenVINO
==========================================

AI-based auto-correction products are becoming increasingly popular due
to their ease of use, editing speed, and affordability. These products
improve the quality of written text in emails, blogs, and chats.

Grammatical Error Correction (GEC) is the task of correcting different
types of errors in text such as spelling, punctuation, grammatical and
word choice errors. GEC is typically formulated as a sentence correction
task. A GEC system takes a potentially erroneous sentence as input and
is expected to transform it into a more correct version. See the example
given below:

=========================== ==========================
Input (Erroneous)           Output (Corrected)
=========================== ==========================
I like to rides my bicycle. I like to ride my bicycle.
=========================== ==========================

As shown in the image below, different types of errors in written
language can be corrected.

.. figure:: https://cdn-images-1.medium.com/max/540/1*Voez5hEn5MU8Knde3fIZfw.png
   :alt: error_types

   error_types

This tutorial shows how to perform grammatical error correction using
OpenVINO. We will use pre-trained models from the `Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. To simplify the user experience, the `Hugging Face
Optimum <https://huggingface.co/docs/optimum>`__ library is used to
convert the models to OpenVINO™ IR format.

It consists of the following steps:

-  Install prerequisites
-  Download and convert models from a public source using the `OpenVINO
   integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Create an inference pipeline for grammatical error checking
-  Optimize grammar correction pipeline with
   `NNCF <https://github.com/openvinotoolkit/nncf/>`__ quantization
-  Compare original and optimized pipelines from performance and
   accuracy standpoints

**Table of contents:**

-  `How does it work?
   <#how-does-it-work>`__
-  `Prerequisites
   <#prerequisites>`__
-  `Download and Convert Models
   <#download-and-convert-models>`__

   -  `Select inference device
      <#select-inference-device>`__
   -  `Grammar Checker
      <#grammar-checker>`__
   -  `Grammar Corrector
      <#grammar-corrector>`__

-  `Prepare Demo Pipeline
   <#prepare-demo-pipeline>`__
-  `Quantization
   <#quantization>`__

   -  `Run Quantization
      <#run-quantization>`__
   -  `Compare model size, performance and accuracy
      <#compare-model-size-performance-and-accuracy>`__

-  `Interactive demo
   <#interactive-demo>`__

How does it work? 
------------------------------------------------------------



A Grammatical Error Correction task can be thought of as a
sequence-to-sequence task where a model is trained to take a
grammatically incorrect sentence as input and return a grammatically
correct sentence as output. We will use the
`FLAN-T5 <https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis>`__
model finetuned on an expanded version of the
`JFLEG <https://paperswithcode.com/dataset/jfleg>`__ dataset.

The version of FLAN-T5 released with the `Scaling Instruction-Finetuned
Language Models <https://arxiv.org/pdf/2210.11416.pdf>`__ paper is an
enhanced version of `T5 <https://huggingface.co/t5-large>`__ that has
been finetuned on a combination of tasks. The paper explores instruction
finetuning with a particular focus on scaling the number of tasks,
scaling the model size, and finetuning on chain-of-thought data. The
paper discovers that overall instruction finetuning is a general method
that improves the performance and usability of pre-trained language
models.

.. figure:: https://production-media.paperswithcode.com/methods/a04cb14e-e6b8-449e-9487-bc4262911d74.png
   :alt: flan-t5_training

   flan-t5_training

For more details about the model, please check out
`paper <https://arxiv.org/abs/2210.11416>`__, original
`repository <https://github.com/google-research/t5x>`__, and Hugging
Face `model card <https://huggingface.co/google/flan-t5-large>`__

Additionally, to reduce the number of sentences required to be
processed, you can perform grammatical correctness checking. This task
should be considered as a simple binary text classification, where the
model gets input text and predicts label 1 if a text contains any
grammatical errors and 0 if it does not. You will use the
`roberta-base-CoLA <https://huggingface.co/textattack/roberta-base-CoLA>`__
model, the RoBERTa Base model finetuned on the CoLA dataset. The RoBERTa
model was proposed in `RoBERTa: A Robustly Optimized BERT Pretraining
Approach paper <https://arxiv.org/abs/1907.11692>`__. It builds on BERT
and modifies key hyperparameters, removing the next-sentence
pre-training objective and training with much larger mini-batches and
learning rates. Additional details about the model can be found in a
`blog
post <https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/>`__
by Meta AI and in the `Hugging Face
documentation <https://huggingface.co/docs/transformers/model_doc/roberta>`__

Now that we know more about FLAN-T5 and RoBERTa, let us get started. 🚀

Prerequisites 
--------------------------------------------------------



First, we need to install the `Hugging Face
Optimum <https://huggingface.co/docs/transformers/index>`__ library
accelerated by OpenVINO integration. The Hugging Face Optimum API is a
high-level API that enables us to convert and quantize models from the
Hugging Face Transformers library to the OpenVINO™ IR format. For more
details, refer to the `Hugging Face Optimum
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.

.. code:: ipython3

    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" "openvino>=2023.1.0" onnx gradio "transformers>=4.33.0"
    %pip install -q "git+https://github.com/openvinotoolkit/nncf.git@9c671f0ae0a118e4bc2de8b09e66425931c0bfa4" datasets jiwer


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Download and Convert Models 
----------------------------------------------------------------------



Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. The Optimum Inference models are API compatible with Hugging
Face Transformers models. This means we just need to replace
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

Below is an example of the RoBERTa text classification model

.. code:: diff

   -from transformers import AutoModelForSequenceClassification
   +from optimum.intel.openvino import OVModelForSequenceClassification
   from transformers import AutoTokenizer, pipeline

   model_id = "textattack/roberta-base-CoLA"
   -model = AutoModelForSequenceClassification.from_pretrained(model_id)
   +model = OVModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)

Model class initialization starts with calling ``from_pretrained``
method. When downloading and converting Transformers model, the
parameter ``from_transformers=True`` should be added. We can save the
converted model for the next usage with the ``save_pretrained`` method.
Tokenizer class and pipelines API are compatible with Optimum models.

.. code:: ipython3

    from pathlib import Path
    from transformers import pipeline, AutoTokenizer
    from optimum.intel.openvino import OVModelForSeq2SeqLM, OVModelForSequenceClassification


.. parsed-literal::

    2023-09-27 14:53:36.462575: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-09-27 14:53:36.496914: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-09-27 14:53:37.063292: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda-11.7'
    /home/nsavel/venvs/ov_notebooks_tmp/lib/python3.8/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
      warnings.warn(


Select inference device 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    import openvino as ov
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Grammar Checker 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    grammar_checker_model_id = "textattack/roberta-base-CoLA"
    grammar_checker_dir = Path("roberta-base-cola")
    grammar_checker_tokenizer = AutoTokenizer.from_pretrained(grammar_checker_model_id)
    
    if grammar_checker_dir.exists():
        grammar_checker_model = OVModelForSequenceClassification.from_pretrained(grammar_checker_dir, device=device.value)
    else:
        grammar_checker_model = OVModelForSequenceClassification.from_pretrained(grammar_checker_model_id, export=True, device=device.value)
        grammar_checker_model.save_pretrained(grammar_checker_dir)


.. parsed-literal::

    Framework not specified. Using pt to export to ONNX.
    Some weights of the model checkpoint at textattack/roberta-base-CoLA were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> False


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    Compiling the model to CPU ...
    Set CACHE_DIR to /tmp/tmpcqv99eqb/model_cache


Let us check model work, using inference pipeline for
``text-classification`` task. You can find more information about usage
Hugging Face inference pipelines in this
`tutorial <https://huggingface.co/docs/transformers/pipeline_tutorial>`__

.. code:: ipython3

    input_text = "They are moved by salar energy"
    grammar_checker_pipe = pipeline("text-classification", model=grammar_checker_model, tokenizer=grammar_checker_tokenizer)
    result = grammar_checker_pipe(input_text)[0]
    print(f"input text: {input_text}")
    print(f'predicted label: {"contains_errors" if result["label"] == "LABEL_1" else "no errors"}')
    print(f'predicted score: {result["score"] :.2}')


.. parsed-literal::

    input text: They are moved by salar energy
    predicted label: contains_errors
    predicted score: 0.88


Great! Looks like the model can detect errors in the sample.

Grammar Corrector 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The steps for loading the Grammar Corrector model are very similar,
except for the model class that is used. Because FLAN-T5 is a
sequence-to-sequence text generation model, we should use the
``OVModelForSeq2SeqLM`` class and the ``text2text-generation`` pipeline
to run it.

.. code:: ipython3

    grammar_corrector_model_id = "pszemraj/flan-t5-large-grammar-synthesis"
    grammar_corrector_dir = Path("flan-t5-large-grammar-synthesis")
    grammar_corrector_tokenizer = AutoTokenizer.from_pretrained(grammar_corrector_model_id)
    
    if grammar_corrector_dir.exists():
        grammar_corrector_model = OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_dir, device=device.value)
    else:
        grammar_corrector_model = OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_model_id, export=True, device=device.value)
        grammar_corrector_model.save_pretrained(grammar_corrector_dir)


.. parsed-literal::

    Framework not specified. Using pt to export to ONNX.
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> False
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> True
    /home/nsavel/venvs/ov_notebooks_tmp/lib/python3.8/site-packages/transformers/modeling_utils.py:875: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if causal_mask.shape[1] < attention_mask.shape[1]:
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> True
    /home/nsavel/venvs/ov_notebooks_tmp/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py:509: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif past_key_value.shape[2] != key_value_states.shape[1]:
    Compiling the encoder to AUTO ...
    Compiling the decoder to AUTO ...
    Compiling the decoder to AUTO ...


.. code:: ipython3

    grammar_corrector_pipe = pipeline("text2text-generation", model=grammar_corrector_model, tokenizer=grammar_corrector_tokenizer)

.. code:: ipython3

    result = grammar_corrector_pipe(input_text)[0]
    print(f"input text:     {input_text}") 
    print(f'generated text: {result["generated_text"]}') 


.. parsed-literal::

    /home/nsavel/venvs/ov_notebooks_tmp/lib/python3.8/site-packages/optimum/intel/openvino/modeling_seq2seq.py:339: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      last_hidden_state = torch.from_numpy(self.request(inputs, shared_memory=True)["last_hidden_state"]).to(
    /home/nsavel/venvs/ov_notebooks_tmp/lib/python3.8/site-packages/optimum/intel/openvino/modeling_seq2seq.py:416: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      self.request.start_async(inputs, shared_memory=True)


.. parsed-literal::

    input text:     They are moved by salar energy
    generated text: They are powered by solar energy.


Nice! The result looks pretty good!

Prepare Demo Pipeline 
----------------------------------------------------------------



Now let us put everything together and create the pipeline for grammar
correction. The pipeline accepts input text, verifies its correctness,
and generates the correct version if required. It will consist of
several steps:

1. Split text on sentences.
2. Check grammatical correctness for each sentence using Grammar
   Checker.
3. Generate an improved version of the sentence if required.

.. code:: ipython3

    import re
    import transformers
    from tqdm.notebook import tqdm
    
    
    def split_text(text: str) -> list:
        """
        Split a string of text into a list of sentence batches.
    
        Parameters:
        text (str): The text to be split into sentence batches.
    
        Returns:
        list: A list of sentence batches. Each sentence batch is a list of sentences.
        """
        # Split the text into sentences using regex
        sentences = re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", text)
    
        # Initialize a list to store the sentence batches
        sentence_batches = []
    
        # Initialize a temporary list to store the current batch of sentences
        temp_batch = []
    
        # Iterate through the sentences
        for sentence in sentences:
            # Add the sentence to the temporary batch
            temp_batch.append(sentence)
    
            # If the length of the temporary batch is between 2 and 3 sentences, or if it is the last batch, add it to the list of sentence batches
            if len(temp_batch) >= 2 and len(temp_batch) <= 3 or sentence == sentences[-1]:
                sentence_batches.append(temp_batch)
                temp_batch = []
    
        return sentence_batches
    
    
    def correct_text(text: str, checker: transformers.pipelines.Pipeline, corrector: transformers.pipelines.Pipeline, separator: str = " ") -> str:
        """
        Correct the grammar in a string of text using a text-classification and text-generation pipeline.
    
        Parameters:
        text (str): The inpur text to be corrected.
        checker (transformers.pipelines.Pipeline): The text-classification pipeline to use for checking the grammar quality of the text.
        corrector (transformers.pipelines.Pipeline): The text-generation pipeline to use for correcting the text.
        separator (str, optional): The separator to use when joining the corrected text into a single string. Default is a space character.
    
        Returns:
        str: The corrected text.
        """
        # Split the text into sentence batches
        sentence_batches = split_text(text)
    
        # Initialize a list to store the corrected text
        corrected_text = []
    
        # Iterate through the sentence batches
        for batch in tqdm(
            sentence_batches, total=len(sentence_batches), desc="correcting text.."
        ):
            # Join the sentences in the batch into a single string
            raw_text = " ".join(batch)
    
            # Check the grammar quality of the text using the text-classification pipeline
            results = checker(raw_text)
    
            # Only correct the text if the results of the text-classification are not LABEL_1 or are LABEL_1 with a score below 0.9
            if results[0]["label"] != "LABEL_1" or (
                results[0]["label"] == "LABEL_1" and results[0]["score"] < 0.9
            ):
                # Correct the text using the text-generation pipeline
                corrected_batch = corrector(raw_text)
                corrected_text.append(corrected_batch[0]["generated_text"])
            else:
                corrected_text.append(raw_text)
    
        # Join the corrected text into a single string
        corrected_text = separator.join(corrected_text)
    
        return corrected_text

Let us see it in action.

.. code:: ipython3

    default_text = (
        "Most of the course is about semantic or  content of language but there are also interesting"
        " topics to be learned from the servicefeatures except statistics in characters in documents.At"
        " this point, He introduces herself as his native English speaker and goes on to say that if"
        " you contine to work on social scnce"
    )
    
    corrected_text = correct_text(default_text, grammar_checker_pipe, grammar_corrector_pipe)


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)



.. parsed-literal::

    correcting text..:   0%|          | 0/1 [00:00<?, ?it/s]


.. code:: ipython3

    print(f"input text:     {default_text}\n") 
    print(f'generated text: {corrected_text}')


.. parsed-literal::

    input text:     Most of the course is about semantic or  content of language but there are also interesting topics to be learned from the servicefeatures except statistics in characters in documents.At this point, He introduces herself as his native English speaker and goes on to say that if you contine to work on social scnce
    
    generated text: Most of the course is about the semantic content of language but there are also interesting topics to be learned from the service features except statistics in characters in documents. At this point, she introduces herself as a native English speaker and goes on to say that if you continue to work on social science, you will continue to be successful.


Quantization 
-------------------------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

Grammar checker model takes up a tiny portion of the whole text
correction pipeline so we optimize only the grammar corrector model.
Grammar corrector itself consists of three models: encoder, first call
decoder and decoder with past. The last model’s share of inference
dominates the other ones. Because of this we quantize only it.

The optimization process contains the following steps:

1. Create a calibration dataset for quantization.
2. Run ``nncf.quantize()`` to obtain quantized models.
3. Serialize the ``INT8`` model using ``openvino.save_model()``
   function.

Please select below whether you would like to run quantization to
improve model inference speed.

.. code:: ipython3

    to_quantize = widgets.Checkbox(
        value=True,
        description='Quantization',
        disabled=False,
    )
    
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



Run Quantization 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Below we retrieve the quantized model. Please see ``utils.py`` for
source code. Quantization is relatively time-consuming and will take
some time to complete.

.. code:: ipython3

    from utils import get_quantized_pipeline
    
    grammar_corrector_pipe_fp32 = grammar_corrector_pipe
    grammar_corrector_pipe_int8 = None
    if to_quantize.value:
        quantized_model_path = Path("quantized_decoder_with_past") / "openvino_model.xml"
        grammar_corrector_pipe_int8 = get_quantized_pipeline(grammar_corrector_pipe_fp32, grammar_corrector_tokenizer, core, grammar_corrector_dir,
                                                             quantized_model_path, device.value)



.. parsed-literal::

    Collecting calibration data:   0%|          | 0/10 [00:00<?, ?it/s]


.. parsed-literal::

    /home/nsavel/workspace/openvino_notebooks/notebooks/214-grammar-correction/utils.py:39: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      return original_fn(\*args, \*\*kwargs)



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    Compiling the encoder to AUTO ...
    Compiling the decoder to AUTO ...
    Compiling the decoder to AUTO ...
    Compiling the decoder to AUTO ...


Let’s see correction results. The generated texts for quantized INT8
model and original FP32 model should be almost the same.

.. code:: ipython3

    if to_quantize.value:
        corrected_text_int8 = correct_text(default_text, grammar_checker_pipe, grammar_corrector_pipe_int8)
        print(f"Input text:                   {default_text}\n")
        print(f'Generated text by INT8 model: {corrected_text_int8}')



.. parsed-literal::

    correcting text..:   0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    Input text:                   Most of the course is about semantic or  content of language but there are also interesting topics to be learned from the servicefeatures except statistics in characters in documents.At this point, He introduces herself as his native English speaker and goes on to say that if you contine to work on social scnce
    
    Generated text by INT8 model: Most of the course is about the semantic content of language but there are also interesting topics to be learned from the service features except statistics in characters in documents. At this point, she introduces himself as a native English speaker and goes on to say that if you continue to work on social issues, you will continue to be successful.


Compare model size, performance and accuracy 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



First, we compare file size of ``FP32`` and ``INT8`` models.

.. code:: ipython3

    from utils import calculate_compression_rate
    
    if to_quantize.value:
        model_size_fp32, model_size_int8 = calculate_compression_rate(grammar_corrector_dir / "openvino_decoder_with_past_model.xml", quantized_model_path)


.. parsed-literal::

    Model footprint comparison:
        * FP32 IR model size: 1658150.26 KB
        * INT8 IR model size: 415713.38 KB

Second, we compare two grammar correction pipelines from performance and
accuracy stand points.

Test split of \ `jfleg <https://huggingface.co/datasets/jfleg>`__\ 
dataset is used for testing. One dataset sample consists of a text with
errors as input and several corrected versions as labels. When measuring
accuracy we use mean ``(1 - WER)`` against corrected text versions,
where WER is Word Error Rate metric.

.. code:: ipython3

    from utils import calculate_inference_time_and_accuracy
    
    TEST_SUBSET_SIZE = 50
    
    if to_quantize.value:
        inference_time_fp32, accuracy_fp32 = calculate_inference_time_and_accuracy(grammar_corrector_pipe_fp32, TEST_SUBSET_SIZE)
        print(f"Evaluation results of FP32 grammar correction pipeline. Accuracy: {accuracy_fp32:.2f}%. Time: {inference_time_fp32:.2f} sec.")
        inference_time_int8, accuracy_int8 = calculate_inference_time_and_accuracy(grammar_corrector_pipe_int8, TEST_SUBSET_SIZE)
        print(f"Evaluation results of INT8 grammar correction pipeline. Accuracy: {accuracy_int8:.2f}%. Time: {inference_time_int8:.2f} sec.")
        print(f"Performance speedup: {inference_time_fp32 / inference_time_int8:.3f}")
        print(f"Accuracy drop :{accuracy_fp32 - accuracy_int8:.2f}%.")
        print(f"Model footprint reduction: {model_size_fp32 / model_size_int8:.3f}")



.. parsed-literal::

    Evaluation:   0%|          | 0/50 [00:00<?, ?it/s]


.. parsed-literal::

    Evaluation results of FP32 grammar correction pipeline. Accuracy: 58.04%. Time: 61.03 sec.



.. parsed-literal::

    Evaluation:   0%|          | 0/50 [00:00<?, ?it/s]


.. parsed-literal::

    Evaluation results of INT8 grammar correction pipeline. Accuracy: 57.46%. Time: 42.38 sec.
    Performance speedup: 1.440
    Accuracy drop :0.59%.
    Model footprint reduction: 3.989

Interactive demo \ 
-----------------------------------------------------------------------------------------------------

.. code:: ipython3

    import gradio as gr
    import time
    
    
    def correct(text, quantized, progress=gr.Progress(track_tqdm=True)):
        grammar_corrector = grammar_corrector_pipe_int8 if quantized else grammar_corrector_pipe
        
        start_time = time.perf_counter()
        corrected_text = correct_text(text, grammar_checker_pipe, grammar_corrector)
        end_time = time.perf_counter()
        
        return corrected_text, f"{end_time - start_time:.2f}"
        
    
    def create_demo_block(quantized: bool, show_model_type: bool):
        model_type = (" optimized" if quantized else " original") if show_model_type else ""
        with gr.Row():
            gr.Markdown(f"## Run{model_type} grammar correction pipeline")
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Text")
            with gr.Column():
                output_text = gr.Textbox(label="Correction")
                correction_time = gr.Textbox(label="Time (seconds)")
        with gr.Row():
            gr.Examples(examples=[default_text], inputs=[input_text])
        with gr.Row():
            button = gr.Button(f"Run{model_type}")
            button.click(correct, inputs=[input_text, gr.Number(quantized, visible=False)], outputs=[output_text, correction_time])
    
    
    with gr.Blocks() as demo:
        gr.Markdown("# Interactive demo")
        quantization_is_present = grammar_corrector_pipe_int8 is not None
        create_demo_block(quantized=False, show_model_type=quantization_is_present)
        if quantization_is_present:
            create_demo_block(quantized=True, show_model_type=True)
    
    
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



.. .. raw:: html

..     <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

