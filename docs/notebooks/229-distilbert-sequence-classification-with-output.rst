Sentiment Analysis with OpenVINO™
=================================

**Sentiment analysis** is the use of natural language processing, text
analysis, computational linguistics, and biometrics to systematically
identify, extract, quantify, and study affective states and subjective
information. This notebook demonstrates how to convert and run a
sequence classification model using OpenVINO.

Imports
-------

.. code:: ipython3

    from transformers import DistilBertForSequenceClassification, AutoTokenizer
    import openvino.runtime as ov
    import warnings
    from pathlib import Path
    import numpy as np
    import time
    import torch

Initializing the Model
----------------------

We will use the transformer-based
`distilbert-base-uncased-finetuned-sst-2-english <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
model from Hugging Face.

.. code:: ipython3

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = DistilBertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=checkpoint
    )

Initializing the Tokenizer
--------------------------

Text Preprocessing cleans the text-based input data so it can be fed
into the model.
`Tokenization <https://towardsdatascience.com/tokenization-for-natural-language-processing-a179a891bad4>`__
splits paragraphs and sentences into smaller units that can be more
easily assigned meaning. It involves cleaning the data and assigning
tokens or IDs to the words, so they are represented in a vector space
where similar words have similar vectors. This helps the model
understand the context of a sentence. Here, we will use
`AutoTokenizer <https://huggingface.co/docs/transformers/main_classes/tokenizer>`__
- a pre-trained tokenizer from Hugging Face: .

.. code:: ipython3

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=checkpoint
    )

Convert to ONNX
---------------

**ONNX** is an open format built to represent machine learning models.
ONNX defines a common set of operators - the building blocks of machine
learning and deep learning models - and a common file format to enable
AI developers to use models with a variety of frameworks, tools,
runtimes, and compilers. We need to convert the model from PyTorch to
ONNX. In order to perform the operation, we use a function
`torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#example-alexnet-from-pytorch-to-onnx>`__
to `convert a Hugging Face
model <https://huggingface.co/blog/convert-transformers-to-onnx#export-with-torchonnx-low-level>`__
to its respective ONNX format.

.. code:: ipython3

    onnx_model = "distilbert.onnx"
    MODEL_DIR = "model/"
    MODEL_DIR = f"{MODEL_DIR}"
    onnx_model_path = Path(MODEL_DIR) / onnx_model
    dummy_model_input = tokenizer("This is a sample", return_tensors="pt")
    torch.onnx.export(
        model,
        tuple(dummy_model_input.values()),
        f=onnx_model,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                      'attention_mask': {0: 'batch_size', 1: 'sequence'},
                      'logits': {0: 'batch_size', 1: 'sequence'}},
        do_constant_folding=True,
        opset_version=13,
    )


.. parsed-literal::

    /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py:223: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      mask, torch.tensor(torch.finfo(scores.dtype).min)


Model Optimizer
---------------

`Model
Optimizer <https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__
is a cross-platform command-line tool that facilitates the transition
between training and deployment environments, performs static model
analysis, and adjusts deep learning models for optimal execution on
end-point target devices.

.. code:: ipython3

    optimizer_command = f'mo \
        --input_model {onnx_model} \
        --output_dir {MODEL_DIR} \
        --model_name {checkpoint} \
        --input input_ids,attention_mask \
        --input_shape "[1,128],[1,128]"'
    ! $optimizer_command


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    Check for a new version of Intel(R) Distribution of OpenVINO(TM) toolkit here https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?cid=other&source=prod&campid=ww_2023_bu_IOTG_OpenVINO-2022-3&content=upg_all&medium=organic or on https://github.com/openvinotoolkit/openvino
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/notebooks/229-distilbert-sequence-classification/model/distilbert-base-uncased-finetuned-sst-2-english.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/notebooks/229-distilbert-sequence-classification/model/distilbert-base-uncased-finetuned-sst-2-english.bin


OpenVINO™ Runtime uses the `Infer
Request <https://docs.openvino.ai/latest/openvino_docs_OV_UG_Infer_request.html>`__
mechanism which enables running models on different devices in
asynchronous or synchronous manners. The model graph is sent as an
argument to the OpenVINO API and an inference request is created. The
default inference mode is AUTO but it can be changed according to
requirements and hardware available. You can explore the different
inference modes and their usage `in
documentation. <https://docs.openvino.ai/latest/openvino_docs_Runtime_Inference_Modes_Overview.html>`__

.. code:: ipython3

    warnings.filterwarnings("ignore")
    core = ov.Core()
    ir_model_xml = str((Path(MODEL_DIR) / checkpoint).with_suffix(".xml"))
    compiled_model = core.compile_model(ir_model_xml)
    infer_request = compiled_model.create_infer_request()

.. code:: ipython3

    def softmax(x):
        """
        Defining a softmax function to extract
        the prediction from the output of the IR format
        Parameters: Logits array
        Returns: Probabilities
        """
    
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

Inference
---------

.. code:: ipython3

    def infer(input_text):
        """
        Creating a generic inference function
        to read the input and infer the result
        into 2 classes: Positive or Negative.
        Parameters: Text to be processed
        Returns: Label: Positive or Negative.
        """
    
        input_text = tokenizer(
            input_text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="np",
        )
        inputs = dict(input_text)
        label = {0: "NEGATIVE", 1: "POSITIVE"}
        result = infer_request.infer(inputs=inputs)
        for i in result.values():
            probability = np.argmax(softmax(i))
        return label[probability]

For a single input sentence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    input_text = "I had a wonderful day"
    start_time = time.perf_counter()
    result = infer(input_text)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("Label: ", result)
    print("Total Time: ", "%.2f" % total_time, " seconds")


.. parsed-literal::

    Label:  POSITIVE
    Total Time:  0.02  seconds


Read from a text file
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    start_time = time.perf_counter()
    with open("../data/text/food_reviews.txt", "r") as f:
        input_text = f.readlines()
        for lines in input_text:
            print("User Input: ", lines)
            result = infer(lines)
            print("Label: ", result, "\n")
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("Total Time: ", "%.2f" % total_time, " seconds")


.. parsed-literal::

    User Input:  The food was horrible.
    
    Label:  NEGATIVE 
    
    User Input:  We went because the restaurant had good reviews.
    Label:  POSITIVE 
    
    Total Time:  0.04  seconds

