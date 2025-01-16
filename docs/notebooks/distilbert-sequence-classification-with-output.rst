Sentiment Analysis with OpenVINO™
=================================

**Sentiment analysis** is the use of natural language processing, text
analysis, computational linguistics, and biometrics to systematically
identify, extract, quantify, and study affective states and subjective
information. This notebook demonstrates how to convert and run a
sequence classification model using OpenVINO.


**Table of contents:**


-  `Imports <#imports>`__
-  `Initializing the Model <#initializing-the-model>`__
-  `Initializing the Tokenizer <#initializing-the-tokenizer>`__
-  `Convert Model to OpenVINO Intermediate Representation
   format <#convert-model-to-openvino-intermediate-representation-format>`__

   -  `Select inference device <#select-inference-device>`__

-  `Inference <#inference>`__

   -  `For a single input sentence <#for-a-single-input-sentence>`__
   -  `Read from a text file <#read-from-a-text-file>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Imports
-------



.. code:: ipython3

    %pip install "openvino>=2023.1.0" transformers "torch>=2.1" tqdm --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cpu
    Requirement already satisfied: openvino>=2023.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.4.0)
    Requirement already satisfied: transformers in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (4.46.1)
    Requirement already satisfied: torch>=2.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2.4.1+cpu)
    Requirement already satisfied: tqdm in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (4.66.6)
    Requirement already satisfied: numpy<2.1.0,>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2023.1.0) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2023.1.0) (2024.1.0)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2023.1.0) (24.1)
    Requirement already satisfied: filelock in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from transformers) (3.16.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from transformers) (0.26.2)
    Requirement already satisfied: pyyaml>=5.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from transformers) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from transformers) (2024.9.11)
    Requirement already satisfied: requests in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from transformers) (2.32.3)
    Requirement already satisfied: safetensors>=0.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from transformers) (0.4.5)
    Requirement already satisfied: tokenizers<0.21,>=0.20 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from transformers) (0.20.2)
    Requirement already satisfied: typing-extensions>=4.8.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (4.12.2)
    Requirement already satisfied: sympy in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (1.13.3)
    Requirement already satisfied: networkx in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (3.1)
    Requirement already satisfied: jinja2 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (3.1.4)
    Requirement already satisfied: fsspec in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (2024.9.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jinja2->torch>=2.1) (2.1.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->transformers) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->transformers) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->transformers) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->transformers) (2024.8.30)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from sympy->torch>=2.1) (1.3.0)
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import warnings
    from pathlib import Path
    import time
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import numpy as np
    import openvino as ov

.. code:: ipython3

    # Fetch `notebook_utils` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, device_widget

Initializing the Model
----------------------



We will use the transformer-based `DistilBERT base uncased finetuned
SST-2 <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
model from Hugging Face.

.. code:: ipython3

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=checkpoint)


.. parsed-literal::

    2024-11-04 23:18:47.102633: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-04 23:18:47.135966: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-11-04 23:18:47.793551: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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
- a pre-trained tokenizer from Hugging Face:

.. code:: ipython3

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint)

Convert Model to OpenVINO Intermediate Representation format
------------------------------------------------------------



`Model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
facilitates the transition between training and deployment environments,
performs static model analysis, and adjusts deep learning models for
optimal execution on end-point target devices.

.. code:: ipython3

    import torch
    
    ir_xml_name = checkpoint + ".xml"
    MODEL_DIR = "model/"
    ir_xml_path = Path(MODEL_DIR) / ir_xml_name
    
    MAX_SEQ_LENGTH = 128
    input_info = [
        (ov.PartialShape([1, -1]), ov.Type.i64),
        (ov.PartialShape([1, -1]), ov.Type.i64),
    ]
    default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)
    inputs = {
        "input_ids": default_input,
        "attention_mask": default_input,
    }
    
    ov_model = ov.convert_model(model, input=input_info, example_input=inputs)
    ov.save_model(ov_model, ir_xml_path)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:5006: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.


OpenVINO™ Runtime uses the `Infer
Request <https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/inference-request.html>`__
mechanism which enables running models on different devices in
asynchronous or synchronous manners. The model graph is sent as an
argument to the OpenVINO API and an inference request is created. The
default inference mode is AUTO but it can be changed according to
requirements and hardware available. You can explore the different
inference modes and their usage `in
documentation. <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html>`__

.. code:: ipython3

    core = ov.Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    warnings.filterwarnings("ignore")
    compiled_model = core.compile_model(ov_model, device.value)
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
    Total Time:  0.03  seconds


Read from a text file
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Download the text from the openvino_notebooks storage
    vocab_file_path = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/text/food_reviews.txt",
        directory="data",
    )



.. parsed-literal::

    data/food_reviews.txt:   0%|          | 0.00/71.0 [00:00<?, ?B/s]


.. code:: ipython3

    start_time = time.perf_counter()
    with vocab_file_path.open(mode="r") as f:
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
    
    Total Time:  0.03  seconds

