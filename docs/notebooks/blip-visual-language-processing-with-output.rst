Visual Question Answering and Image Captioning using BLIP and OpenVINO
======================================================================

Humans perceive the world through vision and language. A longtime goal
of AI is to build intelligent agents that can understand the world
through vision and language inputs to communicate with humans through
natural language. In order to achieve this goal, vision-language
pre-training has emerged as an effective approach, where deep neural
network models are pre-trained on large scale image-text datasets to
improve performance on downstream vision-language tasks, such as
image-text retrieval, image captioning, and visual question answering.

`BLIP <https://github.com/salesforce/BLIP>`__ is a language-image
pre-training framework for unified vision-language understanding and
generation. BLIP achieves state-of-the-art results on a wide range of
vision-language tasks. This tutorial demonstrates how to use BLIP for
visual question answering and image captioning. An additional part of
tutorial demonstrates how to speed up the model by applying 8-bit
post-training quantization and data free int8 weight compression from
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ (Neural Network
Compression Framework) to OpenVINO IR models and infer optimized BLIP
model via OpenVINO™ Toolkit.

The tutorial consists of the following parts:

1. Instantiate a BLIP model.
2. Convert the BLIP model to OpenVINO IR.
3. Run visual question answering and image captioning with OpenVINO.
4. Optimize BLIP model using NNCF
5. Compare original and optimized models
6. Launch interactive demo


**Table of contents:**


-  `Background <#background>`__

   -  `Image Captioning <#image-captioning>`__
   -  `Visual Question Answering <#visual-question-answering>`__

-  `Instantiate Model <#instantiate-model>`__
-  `Convert Models to OpenVINO IR <#convert-models-to-openvino-ir>`__

   -  `Vision Model <#vision-model>`__
   -  `Text Encoder <#text-encoder>`__
   -  `Text Decoder <#text-decoder>`__

-  `Run OpenVINO Model <#run-openvino-model>`__

   -  `Prepare Inference Pipeline <#prepare-inference-pipeline>`__
   -  `Select inference device <#select-inference-device>`__
   -  `Image Captioning <#image-captioning>`__
   -  `Question Answering <#question-answering>`__

-  `Optimize model using NNCF <#optimize-model-using-nncf>`__

   -  `Prepare dataset <#prepare-dataset>`__
   -  `Quantize vision model <#quantize-vision-model>`__
   -  `Quantize text encoder <#quantize-text-encoder>`__
   -  `Compress weights of text
      decoder <#compress-weights-of-text-decoder>`__
   -  `Run optimized OpenVINO model <#run-optimized-openvino-model>`__

      -  `Image captioning <#image-captioning>`__
      -  `Question answering <#question-answering>`__

   -  `Compare file sizes <#compare-file-sizes>`__
   -  `Compare inference time of the FP16 and optimized
      models <#compare-inference-time-of-the-fp16-and-optimized-models>`__

-  `Interactive demo <#interactive-demo>`__



This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Background
----------



Visual language processing is a branch of artificial intelligence that
focuses on creating algorithms designed to enable computers to more
accurately understand images and their content.

Popular tasks include:

-  **Text to Image Retrieval** - a semantic task that aims to find the
   most relevant image for a given text description.
-  **Image Captioning** - a semantic task that aims to provide a text
   description for image content.
-  **Visual Question Answering** - a semantic task that aims to answer
   questions based on image content.

As shown in the diagram below, these three tasks differ in the input
provided to the AI system. For text-to-image retrieval, you have a
predefined gallery of images for search and a user-requested text
description (query). Image captioning can be represented as a particular
case of visual question answering, where you have a predefined question
“What is in the picture?” and various images provided by a user. For
visual question answering, both the text-based question and image
context are variables requested by a user.

|image0|

This notebook does not focus on Text to Image retrieval. Instead, it
considers Image Captioning and Visual Question Answering.

Image Captioning
~~~~~~~~~~~~~~~~



Image Captioning is the task of describing the content of an image in
words. This task lies at the intersection of computer vision and natural
language processing. Most image captioning systems use an
encoder-decoder framework, where an input image is encoded into an
intermediate representation of the information in the image, and then
decoded into a descriptive text sequence.

|image1|

Visual Question Answering
~~~~~~~~~~~~~~~~~~~~~~~~~



Visual Question Answering (VQA) is the task of answering text-based
questions about image content.

|image2|

For a better understanding of how VQA works, let us consider a
traditional NLP task like Question Answering, which aims to retrieve the
answer to a question from a given text input. Typically, a question
answering pipeline consists of three steps:

|image3|

1. Question analysis - analysis of provided question in natural language
   form to understand the object in the question and additional context.
   For example, if you have a question like “How many bridges in
   Paris?”, question words *“how many”* gives a hint that the answer is
   more likely to be a number, *“bridges”* is the target object of the
   question and *" in Paris"* serves as additional context for the
   search.
2. Build query for search - use analyzed results to formalize query for
   finding the most relevant information.
3. Perform a search in the knowledge base - send the query to a
   knowledge base, typically provided text documents or databases serve
   as a source of knowledge.

|image4|

The difference between text-based question answering and visual question
answering is that an image is used as context and the knowledge base.

|image5|

Answering arbitrary questions about images is a complex problem because
it requires involving a lot of computer vision sub-tasks. In the table
below, you can find an example of questions and the required computer
vision skills to find answers.

+-----------------------------+----------------------------------------+
| Computer vision task        | Question examples                      |
+=============================+========================================+
| Object recognition          | What is shown in the picture? What is  |
|                             | it?                                    |
+-----------------------------+----------------------------------------+
| Object detection            | Is there any object (dog, man, book)   |
|                             | in the image? Where is … located?      |
+-----------------------------+----------------------------------------+
| Object and image attribute  | What color is an umbrella? Does this   |
| recognition                 | man wear glasses? Is there color in    |
|                             | the image?                             |
+-----------------------------+----------------------------------------+
| Scene recognition           | Is it rainy? What celebration is       |
|                             | pictured?                              |
+-----------------------------+----------------------------------------+
| Object counting             | How many players are there on the      |
|                             | football field? How many steps are     |
|                             | there on the stairs?                   |
+-----------------------------+----------------------------------------+
| Activity recognition        | Is the baby crying? What is the woman  |
|                             | cooking? What are they doing?          |
+-----------------------------+----------------------------------------+
| Spatial relationships among | What is located between the sofa and   |
| objects                     | the armchair? What is in the bottom    |
|                             | left corner?                           |
+-----------------------------+----------------------------------------+
| Commonsense reasoning       | Does she have 100% vision? Does this   |
|                             | person have children?                  |
+-----------------------------+----------------------------------------+
| Knowledge-based reasoning   | Is it a vegetarian pizza?              |
+-----------------------------+----------------------------------------+
| Text recognition            | What is the title of the book? What is |
|                             | shown on the screen?                   |
+-----------------------------+----------------------------------------+

There are a lot of applications for visual question answering:

-  Aid Visually Impaired Persons: VQA models can be used to reduce
   barriers for visually impaired people by helping them get information
   about images from the web and the real world.
-  Education: VQA models can be used to improve visitor experiences at
   museums by enabling observers to directly ask questions they are
   interested in or to bring more interactivity to schoolbooks for
   children interested in acquiring specific knowledge.
-  E-commerce: VQA models can retrieve information about products using
   photos from online stores.
-  Independent expert assessment: VQA models can be provide objective
   assessments in sports competitions, medical diagnosis, and forensic
   examination.

.. |image0| image:: https://user-images.githubusercontent.com/29454499/221755717-a5b51b7e-523c-461f-b30c-4edbfaf9a134.png
.. |image1| image:: https://user-images.githubusercontent.com/29454499/221640847-1868117c-aac0-4806-99a4-34f218e98bb8.png
.. |image2| image:: https://user-images.githubusercontent.com/29454499/221641984-3c6d8b2f-dd0d-4302-a4d8-0f8564fca772.png
.. |image3| image:: https://user-images.githubusercontent.com/29454499/221760881-378f1ea8-eadc-4610-aff0-69ecabf62fff.png
.. |image4| image:: https://user-images.githubusercontent.com/29454499/222094861-3cafdf9f-d700-4741-b6c5-fb09c1a4da9a.png
.. |image5| image:: https://user-images.githubusercontent.com/29454499/222095118-3d5826e4-2662-4d1c-abf2-a515f23d6d6a.png

Instantiate Model
-----------------



The BLIP model was proposed in the `BLIP: Bootstrapping Language-Image
Pre-training for Unified Vision-Language Understanding and
Generation <https://arxiv.org/abs/2201.12086>`__ paper.

.. figure:: https://github.com/salesforce/BLIP/raw/main/BLIP.gif
   :alt: blip.gif

   blip.gif

To pre-train a unified vision-language model with both understanding and
generation capabilities, BLIP introduces a multimodal mixture of an
encoder-decoder and a multi-task model which can operate in one of the
three modes:

-  **Unimodal encoders**, which separately encode images and text. The
   image encoder is a vision transformer. The text encoder is the same
   as BERT.
-  **Image-grounded text encoder**, which injects visual information by
   inserting a cross-attention layer between the self-attention layer
   and the feed-forward network for each transformer block of the text
   encoder.
-  **Image-grounded text decoder**, which replaces the bi-directional
   self-attention layers in the text encoder with causal self-attention
   layers.

More details about the model can be found in the `research
paper <https://arxiv.org/abs/2201.12086>`__, `Salesforce
blog <https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/>`__,
`GitHub repo <https://github.com/salesforce/BLIP>`__ and `Hugging Face
model
documentation <https://huggingface.co/docs/transformers/model_doc/blip>`__.

In this tutorial, you will use the
`blip-vqa-base <https://huggingface.co/Salesforce/blip-vqa-base>`__
model available for download from `Hugging
Face <https://huggingface.co/>`__. The same actions are also applicable
to other similar models from the BLIP family. Although this model class
is designed to perform question answering, its components can also be
reused for image captioning.

To start working with the model, you need to instantiate the
``BlipForQuestionAnswering`` class, using ``from_pretrained`` method.
``BlipProcessor`` is a helper class for preparing input data for both
text and vision modalities and postprocessing of generation results.

.. code:: ipython3

    import platform

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0" torchvision "transformers>=4.26.0" "gradio>=4.19" "openvino>=2023.3.0" "datasets>=2.14.6" "nncf>=2.8.1" "tqdm"
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"

.. code:: ipython3

    import time
    from PIL import Image
    from transformers import BlipProcessor, BlipForQuestionAnswering

    # Fetch `notebook_utils` module
    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, device_widget, quantization_widget

    # get model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    # setup test input: download and read image, prepare question
    img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    download_file(img_url, "demo.jpg")
    raw_image = Image.open("demo.jpg").convert("RGB")
    question = "how many dogs are in the picture?"
    # preprocess input data
    inputs = processor(raw_image, question, return_tensors="pt")

    start = time.perf_counter()
    # perform generation
    out = model.generate(**inputs)
    end = time.perf_counter() - start

    # postprocess result
    answer = processor.decode(out[0], skip_special_tokens=True)

.. code:: ipython3

    print(f"Processing time: {end:.4f} s")


.. parsed-literal::

    Processing time: 0.3707 s


.. code:: ipython3

    from pathlib import Path

    if not Path("./utils.py").exists():
        download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/blip-visual-language-processing/utils.py")
    from utils import visualize_results

    fig = visualize_results(raw_image, answer, question)



.. image:: blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_7_0.png


Convert Models to OpenVINO IR
-----------------------------



Starting from OpenVINO 2023.0 release, OpenVINO supports direct PyTorch
models conversion to OpenVINO Intermediate Representation (IR) format to
take the advantage of advanced OpenVINO optimization tools and features.
You need to provide a model object, input data for model tracing to
OpenVINO Model Conversion API. ``ov.convert_model`` function convert
PyTorch model instance to ``ov.Model`` object that can be used for
compilation on device or saved on disk using ``ov.save_model`` in
compressed to FP16 format.

The model consists of three parts:

-  vision_model - an encoder for image representation.
-  text_encoder - an encoder for input query, used for question
   answering and text-to-image retrieval only.
-  text_decoder - a decoder for output answer.

To be able to perform multiple tasks, using the same model components,
you should convert each part independently.

Vision Model
~~~~~~~~~~~~



The vision model accepts float input tensors with the [1,3,384,384]
shape, containing RGB image pixel values normalized in the [0,1] range.

.. code:: ipython3

    import torch
    from pathlib import Path
    import openvino as ov

    VISION_MODEL_OV = Path("blip_vision_model.xml")
    vision_model = model.vision_model
    vision_model.eval()

    # check that model works and save it outputs for reusage as text encoder input
    with torch.no_grad():
        vision_outputs = vision_model(inputs["pixel_values"])

    # if openvino model does not exist, convert it to IR
    if not VISION_MODEL_OV.exists():
        # export pytorch model to ov.Model
        with torch.no_grad():
            ov_vision_model = ov.convert_model(vision_model, example_input=inputs["pixel_values"])
        # save model on disk for next usages
        ov.save_model(ov_vision_model, VISION_MODEL_OV)
        print(f"Vision model successfuly converted and saved to {VISION_MODEL_OV}")
    else:
        print(f"Vision model will be loaded from {VISION_MODEL_OV}")


.. parsed-literal::

    /home/ltalamanova/tmp_venv/lib/python3.11/site-packages/transformers/modeling_utils.py:4225: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


.. parsed-literal::

    Vision model successfuly converted and saved to blip_vision_model.xml


Text Encoder
~~~~~~~~~~~~



The text encoder is used by visual question answering tasks to build a
question embedding representation. It takes ``input_ids`` with a
tokenized question and output image embeddings obtained from the vision
model and attention masks for them.

.. code:: ipython3

    TEXT_ENCODER_OV = Path("blip_text_encoder.xml")


    text_encoder = model.text_encoder
    text_encoder.eval()

    # if openvino model does not exist, convert it to IR
    if not TEXT_ENCODER_OV.exists():
        # prepare example inputs
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
        input_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_attention_mask,
        }
        # export PyTorch model
        with torch.no_grad():
            ov_text_encoder = ov.convert_model(text_encoder, example_input=input_dict)
        # save model on disk for next usages
        ov.save_model(ov_text_encoder, TEXT_ENCODER_OV)
        print(f"Text encoder successfuly converted and saved to {TEXT_ENCODER_OV}")
    else:
        print(f"Text encoder will be loaded from {TEXT_ENCODER_OV}")


.. parsed-literal::

    Text encoder successfuly converted and saved to blip_text_encoder.xml


Text Decoder
~~~~~~~~~~~~



The text decoder is responsible for generating the sequence of tokens to
represent model output (answer to question or caption), using an image
(and question, if required) representation. The generation approach is
based on the assumption that the probability distribution of a word
sequence can be decomposed into the product of conditional next word
distributions. In other words, model predicts the next token in the loop
guided by previously generated tokens until the stop-condition will be
not reached (generated sequence of maximum length or end of string token
obtained). The way the next token will be selected over predicted
probabilities is driven by the selected decoding methodology. You can
find more information about the most popular decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__. The entry point
for the generation process for models from the Hugging Face Transformers
library is the ``generate`` method. You can find more information about
its parameters and configuration in the
`documentation <https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate>`__.
To preserve flexibility in the selection decoding methodology, you will
convert only model inference for one step.

To optimize the generation process and use memory more efficiently, the
``use_cache=True`` option is enabled. Since the output side is
auto-regressive, an output token hidden state remains the same once
computed for every further generation step. Therefore, recomputing it
every time you want to generate a new token seems wasteful. With the
cache, the model saves the hidden state once it has been computed. The
model only computes the one for the most recently generated output token
at each time step, re-using the saved ones for hidden tokens. This
reduces the generation complexity from O(n^3) to O(n^2) for a
transformer model. More details about how it works can be found in this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.
With this option, the model gets the previous step’s hidden states as
input and additionally provides hidden states for the current step as
output. Initially, you have no previous step hidden states, so the first
step does not require you to provide them, but we should initialize them
by default values. In PyTorch, past hidden state outputs are represented
as a list of pairs (hidden state for key, hidden state for value] for
each transformer layer in the model. OpenVINO model does not support
nested outputs, they will be flattened.

Similar to ``text_encoder``, ``text_decoder`` can work with input
sequences of different lengths and requires preserving dynamic input
shapes.

.. code:: ipython3

    text_decoder = model.text_decoder
    text_decoder.eval()

    TEXT_DECODER_OV = Path("blip_text_decoder_with_past.xml")

    # prepare example inputs
    input_ids = torch.tensor([[30522]])  # begin of sequence token id
    attention_mask = torch.tensor([[1]])  # attention mask for input_ids
    encoder_hidden_states = torch.rand((1, 10, 768))  # encoder last hidden state from text_encoder
    encoder_attention_mask = torch.ones((1, 10), dtype=torch.long)  # attention mask for encoder hidden states

    input_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }
    text_decoder_outs = text_decoder(**input_dict)
    # extend input dictionary with hidden states from previous step
    input_dict["past_key_values"] = text_decoder_outs["past_key_values"]

    text_decoder.config.torchscript = True
    if not TEXT_DECODER_OV.exists():
        # export PyTorch model
        with torch.no_grad():
            ov_text_decoder = ov.convert_model(text_decoder, example_input=input_dict)
        # save model on disk for next usages
        ov.save_model(ov_text_decoder, TEXT_DECODER_OV)
        print(f"Text decoder successfuly converted and saved to {TEXT_DECODER_OV}")
    else:
        print(f"Text decoder will be loaded from {TEXT_DECODER_OV}")


.. parsed-literal::

    /home/ltalamanova/tmp_venv/lib/python3.11/site-packages/transformers/models/blip/modeling_blip_text.py:635: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if causal_mask.shape[1] < attention_mask.shape[1]:
    /home/ltalamanova/tmp_venv/lib/python3.11/site-packages/torch/jit/_trace.py:165: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
      if a.grad is not None:


.. parsed-literal::

    Text decoder successfuly converted and saved to blip_text_decoder_with_past.xml


Run OpenVINO Model
------------------



Prepare Inference Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~



As discussed before, the model consists of several blocks which can be
reused for building pipelines for different tasks. In the diagram below,
you can see how image captioning works:

|image01|

The visual model accepts the image preprocessed by ``BlipProcessor`` as
input and produces image embeddings, which are directly passed to the
text decoder for generation caption tokens. When generation is finished,
output sequence of tokens is provided to ``BlipProcessor`` for decoding
to text using a tokenizer.

The pipeline for question answering looks similar, but with additional
question processing. In this case, image embeddings and question
tokenized by ``BlipProcessor`` are provided to the text encoder and then
multimodal question embedding is passed to the text decoder for
performing generation of answers.

|image11|

The next step is implementing both pipelines using OpenVINO models.

.. |image01| image:: https://user-images.githubusercontent.com/29454499/221865836-a56da06e-196d-449c-a5dc-4136da6ab5d5.png
.. |image11| image:: https://user-images.githubusercontent.com/29454499/221868167-d0081add-d9f3-4591-80e7-4753c88c1d0a.png

.. code:: ipython3

    # create OpenVINO Core object instance
    core = ov.Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device = device_widget()

    device


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)




.. parsed-literal::

    Dropdown(description='Device:', index=4, options=('CPU', 'GPU.0', 'GPU.1', 'GPU.2', 'AUTO'), value='AUTO')



.. code:: ipython3

    # load models on device
    ov_vision_model = core.compile_model(VISION_MODEL_OV, device.value)
    ov_text_encoder = core.compile_model(TEXT_ENCODER_OV, device.value)
    ov_text_decoder_with_past = core.compile_model(TEXT_DECODER_OV, device.value)

.. code:: ipython3

    from functools import partial
    from blip_model import text_decoder_forward

    text_decoder.forward = partial(text_decoder_forward, ov_text_decoder_with_past=ov_text_decoder_with_past)

The model helper class has two methods for generation:
**generate_answer** - used for visual question answering,
**generate_caption** - used for caption generation. For initialization,
model class accepts compiled OpenVINO models for the text encoder,
vision model and text decoder, and also configuration for generation and
initial token for decoder work.

.. code:: ipython3

    if not Path("./blip_model.py").exists():
        download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/blip-visual-language-processing/blip_model.py")
    from blip_model import OVBlipModel

    ov_model = OVBlipModel(model.config, model.decoder_start_token_id, ov_vision_model, ov_text_encoder, text_decoder)
    out = ov_model.generate_answer(**inputs, max_length=20)

Now, the model is ready for generation.

Image Captioning
~~~~~~~~~~~~~~~~



.. code:: ipython3

    out = ov_model.generate_caption(inputs["pixel_values"], max_length=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    fig = visualize_results(raw_image, caption)



.. image:: blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_25_0.png


Question Answering
~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    start = time.perf_counter()
    out = ov_model.generate_answer(**inputs, max_length=20)
    end = time.perf_counter() - start
    answer = processor.decode(out[0], skip_special_tokens=True)
    fig = visualize_results(raw_image, answer, question)



.. image:: blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_27_0.png


.. code:: ipython3

    print(f"Processing time: {end:.4f}")


.. parsed-literal::

    Processing time: 0.1186


Optimize model using NNCF
-------------------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding the quantization layers into the
model graph and then using a subset of the training dataset to
initialize the parameters of these additional quantization layers. The
framework is designed so that modifications to your original training
code are minor.

The optimization process contains the following steps:

1. Create a dataset for quantization.
2. Run ``nncf.quantize`` to get a quantized model from the pre-trained
   ``FP16`` model.
3. Serialize the ``INT8`` model using ``openvino.save_model`` function.

..

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time. You can disable
   it using widget below:

.. code:: ipython3

    to_quantize = quantization_widget()

    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



.. code:: ipython3

    VISION_MODEL_OV_INT8 = Path(str(VISION_MODEL_OV).replace(".xml", "_int8.xml"))
    TEXT_ENCODER_OV_INT8 = Path(str(TEXT_ENCODER_OV).replace(".xml", "_int8.xml"))
    TEXT_DECODER_OV_INT8 = Path(str(TEXT_DECODER_OV).replace(".xml", "_int8.xml"))
    int8_model = None

    # Fetch skip_kernel_extension module
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)

    %load_ext skip_kernel_extension

Prepare dataset
~~~~~~~~~~~~~~~



The `VQAv2 <https://visualqa.org/>`__ is a dataset containing
open-ended questions about images. These questions require an
understanding of vision, language and commonsense knowledge to answer.

.. code:: ipython3

    %%skip not $to_quantize.value

    import numpy as np
    from datasets import load_dataset
    from tqdm.notebook import tqdm

    def preprocess_batch(batch, vision_model, inputs_info):
        """
        Preprocesses a dataset batch by loading and transforming image and text data.
        VQAv2 dataset contains multiple questions to image.
        To reduce dataset preparation time we will store preprocessed images in `inputs_info`.
        """
        image_id = batch["image_id"]
        if image_id in inputs_info:
            inputs = processor(text=batch['question'], return_tensors="np")
            pixel_values = inputs_info[image_id]["pixel_values"]
            encoder_hidden_states = inputs_info[image_id]["encoder_hidden_states"]
        else:
            inputs = processor(images=batch["image"], text=batch["question"], return_tensors="np")
            pixel_values = inputs["pixel_values"]
            encoder_hidden_states = vision_model(pixel_values)[vision_model.output(0)]
            inputs_info[image_id] = {
                "pixel_values": pixel_values,
                "encoder_hidden_states": encoder_hidden_states,
                "text_encoder_inputs": []
            }

        text_encoder_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        inputs_info[image_id]["text_encoder_inputs"].append(text_encoder_inputs)


    def prepare_input_data(dataloader, vision_model, opt_init_steps):
        """
        Store calibration subset in List to reduce quantization time.
        """
        inputs_info = {}
        for idx, batch in enumerate(tqdm(dataloader, total=opt_init_steps, desc="Prepare calibration data")):
            preprocess_batch(batch, vision_model, inputs_info)

        calibration_subset = []
        for image_id in inputs_info:
            pixel_values = inputs_info[image_id]["pixel_values"]
            encoder_hidden_states = inputs_info[image_id]["encoder_hidden_states"]
            encoder_attention_mask = np.ones(encoder_hidden_states.shape[:-1], dtype=int)
            for text_encoder_inputs in inputs_info[image_id]["text_encoder_inputs"]:
                text_encoder_inputs["encoder_hidden_states"] = encoder_hidden_states
                text_encoder_inputs["encoder_attention_mask"] = encoder_attention_mask
                blip_inputs = {
                    "vision_model_inputs": {"pixel_values": pixel_values},
                    "text_encoder_inputs": text_encoder_inputs,
                }
                calibration_subset.append(blip_inputs)
        return calibration_subset


    def prepare_dataset(vision_model, opt_init_steps=300, streaming=False):
        """
        Prepares a vision-text dataset for quantization.
        """
        split = f"train[:{opt_init_steps}]" if not streaming else "train"
        dataset = load_dataset("HuggingFaceM4/VQAv2", split=split, streaming=streaming, trust_remote_code=True)
        dataset = dataset.shuffle(seed=42)
        if streaming:
            dataset = dataset.take(opt_init_steps)
        calibration_subset = prepare_input_data(dataset, vision_model, opt_init_steps)
        return calibration_subset

Loading and processing the dataset in streaming mode may take a long
time and depends on your internet connection.

.. code:: ipython3

    %%skip not $to_quantize.value

    import nncf

    comp_vision_model = core.compile_model(VISION_MODEL_OV, device.value)
    calibration_data = prepare_dataset(comp_vision_model)

Quantize vision model
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value

    vision_dataset = nncf.Dataset(calibration_data, lambda x: x["vision_model_inputs"])
    vision_model = core.read_model(VISION_MODEL_OV)

    quantized_model = nncf.quantize(
        model=vision_model,
        calibration_dataset=vision_dataset,
        model_type=nncf.ModelType.TRANSFORMER
    )

    ov.save_model(quantized_model, VISION_MODEL_OV_INT8)



.. parsed-literal::

    Output()


















.. parsed-literal::

    Output()

















.. parsed-literal::

    INFO:nncf:36 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:48 ignored nodes were found by name in the NNCFGraph



.. parsed-literal::

    Output()


















.. parsed-literal::

    Output()

















Quantize text encoder
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value

    text_encoder_dataset = nncf.Dataset(calibration_data, lambda x: x["text_encoder_inputs"])
    text_encoder_model = core.read_model(TEXT_ENCODER_OV)

    quantized_model = nncf.quantize(
        model=text_encoder_model,
        calibration_dataset=text_encoder_dataset,
        model_type=nncf.ModelType.TRANSFORMER
    )
    ov.save_model(quantized_model, TEXT_ENCODER_OV_INT8)



.. parsed-literal::

    Output()


















.. parsed-literal::

    Output()

















.. parsed-literal::

    INFO:nncf:72 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:73 ignored nodes were found by name in the NNCFGraph



.. parsed-literal::

    Output()


















.. parsed-literal::

    Output()

















Compress weights of text decoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The quantization of the text decoder leads to significant accuracy loss.
Instead of post-training quantization, we can use data free weights
compression to reduce the model footprint.

The optimization process contains the following steps:

1. Run ``nncf.compress_weights`` to get a model with compressed weights.
2. Serialize the ``OpenVINO`` model using ``openvino.save_model``
   function.

.. code:: ipython3

    %%skip not $to_quantize.value

    text_decoder_model = core.read_model(TEXT_DECODER_OV)
    compressed_text_decoder = nncf.compress_weights(text_decoder_model)
    ov.save_model(compressed_text_decoder, str(TEXT_DECODER_OV_INT8))


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 100% (124 / 124)          | 100% (124 / 124)                  |
    +--------------+---------------------------+-----------------------------------+



.. parsed-literal::

    Output()

















Run optimized OpenVINO model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The steps for making predictions with the optimized OpenVINO BLIP model
are similar to the PyTorch model. Let us check the model result using
the same input data like for model before quantization

.. code:: ipython3

    %%skip not $to_quantize.value

    q_ov_vision_model = core.compile_model(VISION_MODEL_OV_INT8, device.value)
    q_ov_text_encoder = core.compile_model(TEXT_ENCODER_OV_INT8, device.value)
    q_ov_text_decoder_with_past = core.compile_model(TEXT_DECODER_OV_INT8, device.value)

.. code:: ipython3

    %%skip not $to_quantize.value

    from functools import partial
    from transformers import BlipForQuestionAnswering
    from blip_model import OVBlipModel, text_decoder_forward

    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    text_decoder = model.text_decoder
    text_decoder.eval()

    text_decoder.forward = partial(text_decoder_forward, ov_text_decoder_with_past=q_ov_text_decoder_with_past)
    int8_model = OVBlipModel(model.config, model.decoder_start_token_id, q_ov_vision_model, q_ov_text_encoder, text_decoder)

.. code:: ipython3

    %%skip not $to_quantize.value

    raw_image = Image.open("demo.jpg").convert('RGB')
    question = "how many dogs are in the picture?"
    # preprocess input data
    inputs = processor(raw_image, question, return_tensors="pt")

Image captioning
^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%skip not $to_quantize.value

    out = int8_model.generate_caption(inputs["pixel_values"], max_length=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    fig = visualize_results(raw_image, caption)



.. image:: blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_47_0.png


Question answering
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%skip not $to_quantize.value

    out = int8_model.generate_answer(**inputs, max_length=20)
    answer = processor.decode(out[0], skip_special_tokens=True)
    fig = visualize_results(raw_image, answer, question)



.. image:: blip-visual-language-processing-with-output_files/blip-visual-language-processing-with-output_49_0.png


Compare file sizes
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%skip not $to_quantize.value

    def calculate_compression_rate(ov_model_path):
        fp16_ir_model_size = Path(ov_model_path).with_suffix(".bin").stat().st_size / 1024
        int8_model_path = str(ov_model_path).replace(".xml", "_int8.xml")
        quantized_model_size = Path(int8_model_path).with_suffix(".bin").stat().st_size / 1024
        print(f'{ov_model_path.as_posix().split(".")[0]}')
        print(f"    * FP16 IR model size: {fp16_ir_model_size:.2f} KB")
        print(f"    * INT8 model size: {quantized_model_size:.2f} KB")
        print(f"    * Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")

.. code:: ipython3

    %%skip not $to_quantize.value

    for fp16_path in [VISION_MODEL_OV, TEXT_ENCODER_OV, TEXT_DECODER_OV]:
        calculate_compression_rate(fp16_path)


.. parsed-literal::

    blip_vision_model
        * FP16 IR model size: 168145.70 KB
        * INT8 model size: 84915.48 KB
        * Model compression rate: 1.980
    blip_text_encoder
        * FP16 IR model size: 268087.16 KB
        * INT8 model size: 134676.75 KB
        * Model compression rate: 1.991
    blip_text_decoder_with_past
        * FP16 IR model size: 269303.35 KB
        * INT8 model size: 135302.53 KB
        * Model compression rate: 1.990


Compare inference time of the FP16 and optimized models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To measure the inference performance of the ``FP16`` and ``INT8``
models, we use median inference time on 100 samples of the calibration
dataset. So we can approximately estimate the speed up of the dynamic
quantized models.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications with static shapes.

.. code:: ipython3

    %%skip not $to_quantize.value

    import time
    import torch

    def calculate_inference_time(blip_model, calibration_data, generate_caption):
        inference_time = []
        for inputs in calibration_data:
            pixel_values = torch.from_numpy(inputs["vision_model_inputs"]["pixel_values"])
            input_ids = torch.from_numpy(inputs["text_encoder_inputs"]["input_ids"])
            attention_mask = torch.from_numpy(inputs["text_encoder_inputs"]["attention_mask"])

            start = time.perf_counter()
            if generate_caption:
                _ = blip_model.generate_caption(pixel_values, max_length=20)
            else:
                _ = blip_model.generate_answer(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, max_length=20)
            end = time.perf_counter()
            delta = end - start
            inference_time.append(delta)
        return np.median(inference_time)

.. code:: ipython3

    %%skip not $to_quantize.value

    fp_original_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    fp_text_decoder = fp_original_model.text_decoder
    fp_text_decoder.eval()

    comp_text_encoder = core.compile_model(TEXT_ENCODER_OV, device.value)
    comp_text_decoder_with_past = core.compile_model(TEXT_DECODER_OV, device.value)
    fp_text_decoder.forward = partial(text_decoder_forward, ov_text_decoder_with_past=comp_text_decoder_with_past)
    fp16_model = OVBlipModel(model.config, model.decoder_start_token_id, comp_vision_model, comp_text_encoder, fp_text_decoder)

.. code:: ipython3

    %%skip not $to_quantize.value

    validation_data = calibration_data[:100]

    int8_caption_latency = calculate_inference_time(int8_model, validation_data, generate_caption=True)
    fp16_caption_latency = calculate_inference_time(fp16_model, validation_data, generate_caption=True)

    print(f"Image Captioning speed up: {fp16_caption_latency / int8_caption_latency:.3f}")


.. parsed-literal::

    Image Captioning speed up: 1.254


.. code:: ipython3

    %%skip not $to_quantize.value

    int8_generate_answer_latency = calculate_inference_time(int8_model, validation_data, generate_caption=False)
    fp16_generate_answer_latency = calculate_inference_time(fp16_model, validation_data, generate_caption=False)
    print(f"Question Answering speed up: {fp16_generate_answer_latency / int8_generate_answer_latency:.3f}")


.. parsed-literal::

    Question Answering speed up: 1.715


Interactive demo
----------------



Please select below whether you would like to use the quantized model to
launch the interactive demo.

.. code:: ipython3

    import ipywidgets as widgets

    use_quantized_model = widgets.Checkbox(
        description="Use quantized model",
        value=int8_model is not None,
        disabled=int8_model is None,
    )

    use_quantized_model




.. parsed-literal::

    Checkbox(value=True, description='Use quantized model')



.. code:: ipython3

    import gradio as gr

    ov_model = int8_model if use_quantized_model.value else ov_model


    def generate_answer(img, question):
        if img is None:
            raise gr.Error("Please upload an image or choose one from the examples list")
        start = time.perf_counter()
        inputs = processor(img, question, return_tensors="pt")
        output = ov_model.generate_answer(**inputs, max_length=20) if len(question) else ov_model.generate_caption(inputs["pixel_values"], max_length=20)
        answer = processor.decode(output[0], skip_special_tokens=True)
        elapsed = time.perf_counter() - start
        html = f"<p>Processing time: {elapsed:.4f}</p>"
        return answer, html

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/blip-visual-language-processing/gradio_helper.py"
        )
        open("gradio_helper.py", "w").write(r.text)

    from gradio_helper import make_demo

    demo = make_demo(fn=generate_answer)

    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
