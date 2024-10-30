Typo Detector with OpenVINO™
============================

Typo detection in AI is a process of identifying and correcting
typographical errors in text data using machine learning algorithms. The
goal of typo detection is to improve the accuracy, readability, and
usability of text by identifying and indicating mistakes made during the
writing process. To detect typos, AI-based typo detectors use various
techniques, such as natural language processing (NLP), machine learning
(ML), and deep learning (DL).

A typo detector takes a sentence as an input and identify all
typographical errors such as misspellings and homophone errors.

This tutorial provides how to use the `Typo
Detector <https://huggingface.co/m3hrdadfi/typo-detector-distilbert-en>`__
from the `Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__ library
in the OpenVINO environment to perform the above task.

The model detects typos in a given text with a high accuracy,
performances of which are listed below,

- Precision score of 0.9923
- Recall score of 0.9859
- f1-score of 0.9891

`Source for above
metrics <https://huggingface.co/m3hrdadfi/typo-detector-distilbert-en>`__

These metrics indicate that the model can correctly identify a high
proportion of both correct and incorrect text, minimizing both false
positives and false negatives.

The model has been pretrained on the
`NeuSpell <https://github.com/neuspell/neuspell>`__ dataset.


**Table of contents:**


-  `Imports <#imports>`__
-  `Methods <#methods>`__

   -  `1. Using the Hugging Face Optimum
      library <#1--using-the-hugging-face-optimum-library>`__

      -  `2. Converting the model to OpenVINO
         IR <#2--converting-the-model-to-openvino-ir>`__

   -  `Select inference device <#select-inference-device>`__
   -  `1. Hugging Face Optimum Intel
      library <#1--hugging-face-optimum-intel-library>`__

      -  `Load the model <#load-the-model>`__
      -  `Load the tokenizer <#load-the-tokenizer>`__

   -  `2. Converting the model to OpenVINO
      IR <#2--converting-the-model-to-openvino-ir>`__

      -  `Load the Pytorch model <#load-the-pytorch-model>`__
      -  `Converting to OpenVINO IR <#converting-to-openvino-ir>`__
      -  `Inference <#inference>`__

   -  `Helper Functions <#helper-functions>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    %pip install -q "diffusers>=0.17.1" "openvino>=2023.1.0" "nncf>=2.5.0" "onnx>=1.11.0,!=1.16.2" "transformers>=4.39.0" "torch>=2.4.1" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Imports
~~~~~~~



.. code:: ipython3

    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline,
    )
    from pathlib import Path
    import numpy as np
    import re
    from typing import List, Dict
    import time


.. parsed-literal::

    2024-10-23 05:18:31.129817: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 05:18:31.163652: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 05:18:31.823748: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Methods
~~~~~~~



The notebook provides two methods to run the inference of typo detector
with OpenVINO runtime, so that you can experience both calling the API
of Optimum with OpenVINO Runtime included, and loading models in other
frameworks, converting them to OpenVINO IR format, and running inference
with OpenVINO Runtime.

1. Using the `Hugging Face Optimum <https://huggingface.co/docs/optimum/index>`__ library
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



The Hugging Face Optimum API is a high-level API that allows us to
convert models from the Hugging Face Transformers library to the
OpenVINO™ IR format. Compiled models in OpenVINO IR format can be loaded
using Optimum. Optimum allows the use of optimization on targeted
hardware.

2. Converting the model to OpenVINO IR
''''''''''''''''''''''''''''''''''''''



The Pytorch model is converted to `OpenVINO IR
format <https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__.
This method provides much more insight to how to set up a pipeline from
model loading to model converting, compiling and running inference with
OpenVINO, so that you could conveniently use OpenVINO to optimize and
accelerate inference for other deep-learning models. The optimization of
targeted hardware is also used here.

The following table summarizes the major differences between the two
methods

+-----------------------------------+----------------------------------+
| Method 1                          | Method 2                         |
+===================================+==================================+
| Load models from Optimum, an      | Load model from transformers     |
| extension of transformers         |                                  |
+-----------------------------------+----------------------------------+
| Load the model in OpenVINO IR     | Convert to OpenVINO IR           |
| format on the fly                 |                                  |
+-----------------------------------+----------------------------------+
| Load the compiled model by        | Compile the OpenVINO IR and run  |
| default                           | inference with OpenVINO Runtime  |
+-----------------------------------+----------------------------------+
| Pipeline is created to run        | Manually run inference.          |
| inference with OpenVINO Runtime   |                                  |
+-----------------------------------+----------------------------------+

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

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

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



1. Hugging Face Optimum Intel library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For this method, we need to install the
``Hugging Face Optimum Intel library`` accelerated by OpenVINO
integration.

Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. The Optimum Inference models are API compatible with Hugging
Face Transformers models. This means we need just replace
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

Import required model class

.. code:: ipython3

    from optimum.intel.openvino import OVModelForTokenClassification

Load the model
''''''''''''''



From the ``OVModelForTokenCLassification`` class we will import the
relevant pre-trained model. To load a Transformers model and convert it
to the OpenVINO format on-the-fly, we set ``export=True`` when loading
your model.

.. code:: ipython3

    # The pretrained model we are using
    model_id = "m3hrdadfi/typo-detector-distilbert-en"

    model_dir = Path("optimum_model")

    # Save the model to the path if not existing
    if model_dir.exists():
        model = OVModelForTokenClassification.from_pretrained(model_dir, device=device.value)
    else:
        model = OVModelForTokenClassification.from_pretrained(model_id, export=True, device=device.value)
        model.save_pretrained(model_dir)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/dynamic_graph/wrappers.py:86: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      op1 = operator(\*args, \*\*kwargs)


Load the tokenizer
''''''''''''''''''



Text Preprocessing cleans the text-based input data so it can be fed
into the model. Tokenization splits paragraphs and sentences into
smaller units that can be more easily assigned meaning. It involves
cleaning the data and assigning tokens or IDs to the words, so they are
represented in a vector space where similar words have similar vectors.
This helps the model understand the context of a sentence. We’re making
use of an
`AutoTokenizer <https://huggingface.co/docs/transformers/main_classes/tokenizer>`__
from Hugging Face, which is essentially a pretrained tokenizer.

.. code:: ipython3

    tokenizer = AutoTokenizer.from_pretrained(model_id)

Then we use the inference pipeline for ``token-classification`` task.
You can find more information about usage Hugging Face inference
pipelines in this
`tutorial <https://huggingface.co/docs/transformers/pipeline_tutorial>`__

.. code:: ipython3

    nlp = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="average",
    )

Function to find typos in a sentence and write them to the terminal

.. code:: ipython3

    def show_typos(sentence: str):
        """
        Detect typos from the given sentence.
        Writes both the original input and typo-tagged version to the terminal.

        Arguments:
        sentence -- Sentence to be evaluated (string)
        """

        typos = [sentence[r["start"] : r["end"]] for r in nlp(sentence)]

        detected = sentence
        for typo in typos:
            detected = detected.replace(typo, f"<i>{typo}</i>")

        print("[Input]: ", sentence)
        print("[Detected]: ", detected)
        print("-" * 130)

Let’s run a demo using the Hugging Face Optimum API.

.. code:: ipython3

    sentences = [
        "He had also stgruggled with addiction during his time in Congress .",
        "The review thoroughla assessed all aspects of JLENS SuR and CPG esign maturit and confidence .",
        "Letterma also apologized two his staff for the satyation .",
        "Vincent Jay had earlier won France 's first gold in gthe 10km biathlon sprint .",
        "It is left to the directors to figure out hpw to bring the stry across to tye audience .",
        "I wnet to the park yestreday to play foorball with my fiends, but it statred to rain very hevaily and we had to stop.",
        "My faorite restuarant servs the best spahgetti in the town, but they are always so buzy that you have to make a resrvation in advnace.",
        "I was goig to watch a mvoie on Netflx last night, but the straming was so slow that I decided to cancled my subscrpition.",
        "My freind and I went campign in the forest last weekend and saw a beutiful sunst that was so amzing it took our breth away.",
        "I  have been stuying for my math exam all week, but I'm stil not very confidet that I will pass it, because there are so many formuals to remeber.",
    ]

    start = time.time()

    for sentence in sentences:
        show_typos(sentence)

    print(f"Time elapsed: {time.time() - start}")


.. parsed-literal::

    [Input]:  He had also stgruggled with addiction during his time in Congress .
    [Detected]:  He had also <i>stgruggled</i> with addiction during his time in Congress .
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  The review thoroughla assessed all aspects of JLENS SuR and CPG esign maturit and confidence .
    [Detected]:  The review <i>thoroughla</i> assessed all aspects of JLENS SuR and CPG <i>esign maturit</i> and confidence .
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  Letterma also apologized two his staff for the satyation .
    [Detected]:  <i>Letterma</i> also apologized <i>two</i> his staff for the <i>satyation</i> .
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  Vincent Jay had earlier won France 's first gold in gthe 10km biathlon sprint .
    [Detected]:  Vincent Jay had earlier won France 's first gold in <i>gthe</i> 10km biathlon sprint .
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  It is left to the directors to figure out hpw to bring the stry across to tye audience .
    [Detected]:  It is left to the directors to figure out <i>hpw</i> to bring the <i>stry</i> across to <i>tye</i> audience .
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  I wnet to the park yestreday to play foorball with my fiends, but it statred to rain very hevaily and we had to stop.
    [Detected]:  I <i>wnet</i> to the park <i>yestreday</i> to play <i>foorball</i> with my <i>fiends</i>, but it <i>statred</i> to rain very <i>hevaily</i> and we had to stop.
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  My faorite restuarant servs the best spahgetti in the town, but they are always so buzy that you have to make a resrvation in advnace.
    [Detected]:  My <i>faorite restuarant servs</i> the best <i>spahgetti</i> in the town, but they are always so <i>buzy</i> that you have to make a <i>resrvation</i> in <i>advnace</i>.
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  I was goig to watch a mvoie on Netflx last night, but the straming was so slow that I decided to cancled my subscrpition.
    [Detected]:  I was <i>goig</i> to watch a <i>mvoie</i> on <i>Netflx</i> last night, but the <i>straming</i> was so slow that I decided to <i>cancled</i> my <i>subscrpition</i>.
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  My freind and I went campign in the forest last weekend and saw a beutiful sunst that was so amzing it took our breth away.
    [Detected]:  My <i>freind</i> and I went <i>campign</i> in the forest last weekend and saw a <i>beutiful sunst</i> that was so <i>amzing</i> it took our <i>breth</i> away.
    ----------------------------------------------------------------------------------------------------------------------------------
    [Input]:  I  have been stuying for my math exam all week, but I'm stil not very confidet that I will pass it, because there are so many formuals to remeber.
    [Detected]:  I  have been <i>stuying</i> for my math exam all week, but I'm <i>stil</i> not very <i>confidet</i> that I will pass it, because there are so many formuals to <i>remeber</i>.
    ----------------------------------------------------------------------------------------------------------------------------------
    Time elapsed: 0.14796948432922363


2. Converting the model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Load the Pytorch model
''''''''''''''''''''''



Use the ``AutoModelForTokenClassification`` class to load the pretrained
pytorch model.

.. code:: ipython3

    model_id = "m3hrdadfi/typo-detector-distilbert-en"
    model_dir = Path("pytorch_model")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    # Save the model to the path if not existing
    if model_dir.exists():
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_id, config=config)
        model.save_pretrained(model_dir)

Converting to OpenVINO IR
'''''''''''''''''''''''''



.. code:: ipython3

    import openvino as ov

    ov_model_path = Path(model_dir) / "typo_detect.xml"

    dummy_model_input = tokenizer("This is a sample", return_tensors="pt")
    ov_model = ov.convert_model(model, example_input=dict(dummy_model_input))
    ov.save_model(ov_model, ov_model_path)

Inference
'''''''''



OpenVINO™ Runtime Python API is used to compile the model in OpenVINO IR
format. The Core class from the ``openvino`` module is imported first.
This class provides access to the OpenVINO Runtime API. The ``core``
object, which is an instance of the ``Core`` class, represents the API
and it is used to compile the model. The output layer is extracted from
the compiled model as it is needed for inference.

.. code:: ipython3

    core = ov.Core()

    compiled_model = core.compile_model(ov_model, device.value)
    output_layer = compiled_model.output(0)

Helper Functions
~~~~~~~~~~~~~~~~



.. code:: ipython3

    def token_to_words(tokens: List[str]) -> Dict[str, int]:
        """
        Maps the list of tokens to words in the original text.
        Built on the feature that tokens starting with '##' is attached to the previous token as tokens derived from the same word.

        Arguments:
        tokens -- List of tokens

        Returns:
        map_to_words -- Dictionary mapping tokens to words in original text
        """

        word_count = -1
        map_to_words = {}
        for token in tokens:
            if token.startswith("##"):
                map_to_words[token] = word_count
                continue
            word_count += 1
            map_to_words[token] = word_count
        return map_to_words

.. code:: ipython3

    def infer(input_text: str) -> Dict[np.ndarray, np.ndarray]:
        """
        Creating a generic inference function to read the input and infer the result

        Arguments:
        input_text -- The text to be infered (String)

        Returns:
        result -- Resulting list from inference
        """

        tokens = tokenizer(
            input_text,
            return_tensors="np",
        )
        inputs = dict(tokens)
        result = compiled_model(inputs)[output_layer]
        return result

.. code:: ipython3

    def get_typo_indexes(
        result: Dict[np.ndarray, np.ndarray],
        map_to_words: Dict[str, int],
        tokens: List[str],
    ) -> List[int]:
        """
        Given results from the inference and tokens-map-to-words, identifies the indexes of the words with typos.

        Arguments:
        result -- Result from inference (tensor)
        map_to_words -- Dictionary mapping tokens to words (Dictionary)

        Results:
        wrong_words -- List of indexes of words with typos
        """

        wrong_words = []
        c = 0
        result_list = result[0][1:-1]
        for i in result_list:
            prob = np.argmax(i)
            if prob == 1:
                if map_to_words[tokens[c]] not in wrong_words:
                    wrong_words.append(map_to_words[tokens[c]])
            c += 1
        return wrong_words

.. code:: ipython3

    def sentence_split(sentence: str) -> List[str]:
        """
        Split the sentence into words and characters

        Arguments:
        sentence - Sentence to be split (string)

        Returns:
        splitted -- List of words and characters
        """

        splitted = re.split("([',. ])", sentence)
        splitted = [x for x in splitted if x != " " and x != ""]
        return splitted

.. code:: ipython3

    def show_typos(sentence: str):
        """
        Detect typos from the given sentence.
        Writes both the original input and typo-tagged version to the terminal.

        Arguments:
        sentence -- Sentence to be evaluated (string)
        """

        tokens = tokenizer.tokenize(sentence)
        map_to_words = token_to_words(tokens)
        result = infer(sentence)
        typo_indexes = get_typo_indexes(result, map_to_words, tokens)

        sentence_words = sentence_split(sentence)

        typos = [sentence_words[i] for i in typo_indexes]

        detected = sentence
        for typo in typos:
            detected = detected.replace(typo, f"<i>{typo}</i>")

        print("   [Input]: ", sentence)
        print("[Detected]: ", detected)
        print("-" * 130)

Let’s run a demo using the converted OpenVINO IR model.

.. code:: ipython3

    sentences = [
        "He had also stgruggled with addiction during his time in Congress .",
        "The review thoroughla assessed all aspects of JLENS SuR and CPG esign maturit and confidence .",
        "Letterma also apologized two his staff for the satyation .",
        "Vincent Jay had earlier won France 's first gold in gthe 10km biathlon sprint .",
        "It is left to the directors to figure out hpw to bring the stry across to tye audience .",
        "I wnet to the park yestreday to play foorball with my fiends, but it statred to rain very hevaily and we had to stop.",
        "My faorite restuarant servs the best spahgetti in the town, but they are always so buzy that you have to make a resrvation in advnace.",
        "I was goig to watch a mvoie on Netflx last night, but the straming was so slow that I decided to cancled my subscrpition.",
        "My freind and I went campign in the forest last weekend and saw a beutiful sunst that was so amzing it took our breth away.",
        "I  have been stuying for my math exam all week, but I'm stil not very confidet that I will pass it, because there are so many formuals to remeber.",
    ]

    start = time.time()

    for sentence in sentences:
        show_typos(sentence)

    print(f"Time elapsed: {time.time() - start}")


.. parsed-literal::

       [Input]:  He had also stgruggled with addiction during his time in Congress .
    [Detected]:  He had also <i>stgruggled</i> with addiction during his time in Congress .
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  The review thoroughla assessed all aspects of JLENS SuR and CPG esign maturit and confidence .
    [Detected]:  The review <i>thoroughla</i> assessed all aspects of JLENS SuR and CPG <i>esign</i> <i>maturit</i> and confidence .
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  Letterma also apologized two his staff for the satyation .
    [Detected]:  <i>Letterma</i> also apologized <i>two</i> his staff for the <i>satyation</i> .
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  Vincent Jay had earlier won France 's first gold in gthe 10km biathlon sprint .
    [Detected]:  Vincent Jay had earlier won France 's first gold in <i>gthe</i> 10km biathlon sprint .
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  It is left to the directors to figure out hpw to bring the stry across to tye audience .
    [Detected]:  It is left to the directors to figure out <i>hpw</i> to bring the <i>stry</i> across to <i>tye</i> audience .
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  I wnet to the park yestreday to play foorball with my fiends, but it statred to rain very hevaily and we had to stop.
    [Detected]:  I <i>wnet</i> to the park <i>yestreday</i> to play <i>foorball</i> with my <i>fiends</i>, but it <i>statred</i> to rain very <i>hevaily</i> and we had to stop.
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  My faorite restuarant servs the best spahgetti in the town, but they are always so buzy that you have to make a resrvation in advnace.
    [Detected]:  My <i>faorite</i> <i>restuarant</i> <i>servs</i> the best <i>spahgetti</i> in the town, but they are always so <i>buzy</i> that you have to make a <i>resrvation</i> in <i>advnace</i>.
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  I was goig to watch a mvoie on Netflx last night, but the straming was so slow that I decided to cancled my subscrpition.
    [Detected]:  I was <i>goig</i> to watch a <i>mvoie</i> on <i>Netflx</i> last night, but the <i>straming</i> was so slow that I decided to <i>cancled</i> my <i>subscrpition</i>.
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  My freind and I went campign in the forest last weekend and saw a beutiful sunst that was so amzing it took our breth away.
    [Detected]:  My <i>freind</i> and I went <i>campign</i> in the forest last weekend and saw a <i>beutiful</i> <i>sunst</i> that was so <i>amzing</i> it took our <i>breth</i> away.
    ----------------------------------------------------------------------------------------------------------------------------------
       [Input]:  I  have been stuying for my math exam all week, but I'm stil not very confidet that I will pass it, because there are so many formuals to remeber.
    [Detected]:  I  have been <i>stuying</i> for my math exam all week, but I'm <i>stil</i> not very <i>confidet</i> that I will pass it, because there are so many formuals to <i>remeber</i>.
    ----------------------------------------------------------------------------------------------------------------------------------
    Time elapsed: 0.10040116310119629

