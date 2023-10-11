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

**Table of contents:**

- `How does it work? <#how-does-it-work>`__
- `Prerequisites <#prerequisites>`__
- `Download and Convert Models <#download-and-convert-models>`__

  - `Select inference device <#select-inference-device>`__
  - `Grammar Checker <#grammar-checker>`__
  - `Grammar Corrector <#grammar-corrector>`__

- `Prepare Demo Pipeline <#prepare-demo-pipeline>`__

How does it work?
###############################################################################################################################

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
###############################################################################################################################

First, we need to install the `Hugging Face
Optimum <https://huggingface.co/docs/transformers/index>`__ library
accelerated by OpenVINO integration. The Hugging Face Optimum API is a
high-level API that enables us to convert and quantize models from the
Hugging Face Transformers library to the OpenVINO™ IR format. For more
details, refer to the `Hugging Face Optimum
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.

.. code:: ipython3

    !pip install -q "git+https://github.com/huggingface/optimum-intel.git" "openvino>=2023.0.0" onnx onnxruntime gradio


.. parsed-literal::

    
    [notice] A new release of pip is available: 23.1.2 -> 23.2
    [notice] To update, run: pip install --upgrade pip


Download and Convert Models
###############################################################################################################################

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

    2023-07-17 14:43:08.812267: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-07-17 14:43:08.850959: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-07-17 14:43:09.468643: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    comet_ml is installed but `COMET_API_KEY` is not set.


Select inference device
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Select device from dropdown list for running inference using OpenVINO:

.. code:: ipython3

    import ipywidgets as widgets
    from openvino.runtime import Core
    
    core = Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=2, options=('CPU', 'GPU', 'AUTO'), value='AUTO')



Grammar Checker
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

    Compiling the model...
    Set CACHE_DIR to roberta-base-cola/model_cache


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

    Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers
    pip install xformers.


.. parsed-literal::

    input text: They are moved by salar energy
    predicted label: contains_errors
    predicted score: 0.88


Great! Looks like the model can detect errors in the sample.

Grammar Corrector
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

    The argument `from_transformers` is deprecated, and will be removed in optimum 2.0.  Use `export` instead
    Framework not specified. Using pt to export to ONNX.
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> False
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> True
    /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/transformers/modeling_utils.py:850: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if causal_mask.shape[1] < attention_mask.shape[1]:
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> True
    /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py:507: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif past_key_value.shape[2] != key_value_states.shape[1]:
    In-place op on output of tensor.shape. See https://pytorch.org/docs/master/onnx.html#avoid-inplace-operations-when-using-tensor-shape-in-tracing-mode
    In-place op on output of tensor.shape. See https://pytorch.org/docs/master/onnx.html#avoid-inplace-operations-when-using-tensor-shape-in-tracing-mode
    Compiling the encoder...
    Compiling the decoder...
    Compiling the decoder...


.. code:: ipython3

    grammar_corrector_pipe = pipeline("text2text-generation", model=grammar_corrector_model, tokenizer=grammar_corrector_tokenizer)

.. code:: ipython3

    result = grammar_corrector_pipe(input_text)[0]
    print(f"input text:     {input_text}") 
    print(f'generated text: {result["generated_text"]}') 


.. parsed-literal::

    input text:     They are moved by salar energy
    generated text: They are powered by solar energy.


Nice! The result looks pretty good!

Prepare Demo Pipeline
###############################################################################################################################

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


Interactive demo
###############################################################################################################################

.. code:: ipython3

    import gradio as gr
    
    
    def correct(text, _=gr.Progress(track_tqdm=True)):
        return correct_text(text, grammar_checker_pipe, grammar_corrector_pipe)
    
    
    demo = gr.Interface(
        correct,
        gr.Textbox(label="Text"),
        gr.Textbox(label="Correction"),
        examples=[default_text],
        allow_flagging="never",
    )
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
