Sentiment Analysis with OpenVINO™
=================================



**Sentiment analysis** is the use of natural language processing, text
analysis, computational linguistics, and biometrics to systematically
identify, extract, quantify, and study affective states and subjective
information. This notebook demonstrates how to convert and run a
sequence classification model using OpenVINO.

.. _top:

**Table of contents**:

- `Imports <#imports>`__
- `Initializing the Model <#initializing-the-model>`__
- `Initializing the Tokenizer <#initializing-the-tokenizer>`__
- `Convert Model to OpenVINO Intermediate Representation format <#convert-model-to-openvino-intermediate-representation-format>`__

  - `Select inference device <#select-inference-device>`__

- `Inference <#inference>`__

  - `For a single input sentence <#for single -a- -input-sentence>`__
  - `Read from a text file <#read-from-a-text-file>`__

Imports `⇑ <#top>`__
###############################################################################################################################


.. code:: ipython3

    import warnings
    from pathlib import Path
    import time
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import numpy as np
    from openvino.tools import mo
    from openvino.runtime import PartialShape, Type, serialize, Core

Initializing the Model `⇑ <#top>`__
###############################################################################################################################

We will use the transformer-based 
`DistilBERT base uncased finetuned SST-2 <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
model from Hugging Face.

.. code:: ipython3

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=checkpoint
    )

Initializing the Tokenizer `⇑ <#top>`__
###############################################################################################################################


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

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=checkpoint
    )

Convert Model to OpenVINO Intermediate Representation format. `⇑ <#top>`__
###############################################################################################################################

`Model conversion API <https://docs.openvino.ai/2023.1/openvino_docs_model_processing_introduction.html>`__
facilitates the transition between training and deployment environments,
performs static model analysis, and adjusts deep learning models for
optimal execution on end-point target devices.

.. code:: ipython3

    ir_xml_name = checkpoint + ".xml"
    MODEL_DIR = "model/"
    ir_xml_path = Path(MODEL_DIR) / ir_xml_name
    ov_model = mo.convert_model(model, input=[mo.InputCutInfo(shape=PartialShape([1, -1]), type=Type.i64), mo.InputCutInfo(shape=PartialShape([1, -1]), type=Type.i64)])
    serialize(ov_model, ir_xml_path)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py:223: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      mask, torch.tensor(torch.finfo(scores.dtype).min)


OpenVINO™ Runtime uses the `Infer Request <https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_Infer_request.html>`__
mechanism which enables running models on different devices in
asynchronous or synchronous manners. The model graph is sent as an
argument to the OpenVINO API and an inference request is created. The
default inference mode is AUTO but it can be changed according to
requirements and hardware available. You can explore the different
inference modes and their usage `in
documentation. <https://docs.openvino.ai/2023.1/openvino_docs_Runtime_Inference_Modes_Overview.html>`__

.. code:: ipython3

    core = Core()

Select inference device `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Select device from dropdown list for running inference using OpenVINO:

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
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

Inference `⇑ <#top>`__
###############################################################################################################################


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

For a single input sentence `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
    Total Time:  0.04  seconds


Read from a text file `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
    
    Total Time:  0.02  seconds

