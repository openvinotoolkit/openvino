Document Visual Question Answering Using Pix2Struct and OpenVINO™
=================================================================

DocVQA (Document Visual Question Answering) is a research field in
computer vision and natural language processing that focuses on
developing algorithms to answer questions related to the content of a
document represented in image format, like a scanned document,
screenshots, or an image of a text document. Unlike other types of
visual question answering, where the focus is on answering questions
related to images or videos, DocVQA is focused on understanding and
answering questions based on the text and layout of a document. The
questions can be about any aspect of the document text. DocVQA requires
understanding the document’s visual content and the ability to read and
comprehend the text in it.

DocVQA offers several benefits compared to OCR (Optical Character
Recognition) technology: \* Firstly, DocVQA can not only recognize and
extract text from a document, but it can also understand the context in
which the text appears. This means it can answer questions about the
document’s content rather than simply provide a digital version. \*
Secondly, DocVQA can handle documents with complex layouts and
structures, like tables and diagrams, which can be challenging for
traditional OCR systems. \* Finally, DocVQA can automate many
document-based workflows, like document routing and approval processes,
to make employees focus on more meaningful work. The potential
applications of DocVQA include automating tasks like information
retrieval, document analysis, and document summarization.

`Pix2Struct <https://arxiv.org/pdf/2210.03347.pdf>`__ is a multimodal
model for understanding visually situated language that easily copes
with extracting information from images. The model is trained using the
novel learning technique to parse masked screenshots of web pages into
simplified HTML, providing a significantly well-suited pretraining data
source for the range of downstream activities such as OCR, visual
question answering, and image captioning.

In this tutorial, we consider how to run the Pix2Struct model using
OpenVINO for solving document visual question answering task. We will
use a pre-trained model from the `Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. To simplify the user experience, the `Hugging Face
Optimum <https://huggingface.co/docs/optimum>`__ library is used to
convert the model to OpenVINO™ IR format.

**Table of contents:**

-  `About Pix2Struct <#about-pixstruct>`__
-  `Prerequisites <#prerequisites>`__
-  `Download and Convert Model <#download-and-convert-model>`__
-  `Select inference device <#select-inference-device>`__
-  `Test model inference <#test-model-inference>`__
-  `Interactive demo <#interactive-demo>`__

About Pix2Struct
----------------



Pix2Struct is an image encoder - text decoder model that is trained on
image-text pairs for various tasks, including image captioning and
visual question answering. The model combines the simplicity of purely
pixel-level inputs with the generality and scalability provided by
self-supervised pretraining from diverse and abundant web data. The
model does this by recommending a screenshot parsing objective that
needs predicting an HTML-based parse from a screenshot of a web page
that has been partially masked. With the diversity and complexity of
textual and visual elements found on the web, Pix2Struct learns rich
representations of the underlying structure of web pages, which can
effectively transfer to various downstream visual language understanding
tasks.

Pix2Struct is based on the Vision Transformer (ViT), an
image-encoder-text-decoder model with changes in input representation to
make the model more robust to processing images with various aspect
ratios. Standard ViT extracts fixed-size patches after scaling input
images to a predetermined resolution. This distorts the proper aspect
ratio of the image, which can be highly variable for documents, mobile
UIs, and figures. Pix2Struct proposes to scale the input image up or
down to extract the maximum number of patches that fit within the given
sequence length. This approach is more robust to extreme aspect ratios,
common in the domains Pix2Struct experiments with. Additionally, the
model can handle on-the-fly changes to the sequence length and
resolution. To handle variable resolutions unambiguously, 2-dimensional
absolute positional embeddings are used for the input patches.

Prerequisites
-------------



First, we need to install the `Hugging Face
Optimum <https://huggingface.co/docs/transformers/index>`__ library
accelerated by OpenVINO integration. The Hugging Face Optimum API is a
high-level API that enables us to convert and quantize models from the
Hugging Face Transformers library to the OpenVINO™ IR format. For more
details, refer to the `Hugging Face Optimum
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.

.. code:: ipython3

    %pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" "openvino>=2023.1.0" "transformers>=4.33.0" onnx gradio

Download and Convert Model
--------------------------



Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. The Optimum Inference models are API compatible with Hugging
Face Transformers models. This means we just need to replace the
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

Model class initialization starts with calling the ``from_pretrained``
method. When downloading and converting the Transformers model, the
parameter ``export=True`` should be added. We can save the converted
model for the next usage with the ``save_pretrained`` method. After
model saving using the ``save_pretrained`` method, you can load your
converted model without the ``export`` parameter, avoiding model
conversion for the next time. For reducing memory consumption, we can
compress model to float16 using ``half()`` method.

In this tutorial, we separate model export and loading for a
demonstration of how to work with the model in both modes. We will use
the
`pix2struct-docvqa-base <https://huggingface.co/google/pix2struct-docvqa-base>`__
model as an example in this tutorial, but the same steps for running are
applicable for other models from pix2struct family.

.. code:: ipython3

    import gc
    from pathlib import Path
    from optimum.intel.openvino import OVModelForPix2Struct
    
    model_id = "google/pix2struct-docvqa-base"
    model_dir = Path(model_id.split('/')[-1])
    
    if not model_dir.exists():
        ov_model = OVModelForPix2Struct.from_pretrained(model_id, export=True, compile=False)
        ov_model.half()
        ov_model.save_pretrained(model_dir)
        del ov_model
        gc.collect();


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    2023-10-20 13:49:09.525682: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-10-20 13:49:09.565139: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-20 13:49:10.397504: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
      warnings.warn(


Select inference device
-----------------------



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    import openvino as ov
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=[d for d in core.available_devices if "GPU" not in d] + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Test model inference
--------------------



The diagram below demonstrates how the model works:
|pix2struct_diagram.png|

For running model inference we should preprocess data first.
``Pix2StructProcessor`` is responsible for preparing input data and
decoding output for the original PyTorch model and easily can be reused
for running with the Optimum Intel model. Then
``OVModelForPix2Struct.generate`` method will launch answer generation.
Finally, generated answer token indices should be decoded in text format
by ``Pix2StructProcessor.decode``

.. |pix2struct_diagram.png| image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/c7456b17-0687-4aa9-851b-267bff3dac79

.. code:: ipython3

    from transformers import Pix2StructProcessor
    
    processor = Pix2StructProcessor.from_pretrained(model_id)
    ov_model = OVModelForPix2Struct.from_pretrained(model_dir, device=device.value)


.. parsed-literal::

    Compiling the encoder to AUTO ...
    Compiling the decoder to AUTO ...
    Compiling the decoder to AUTO ...


Let’s see the model in action. For testing the model, we will use a
screenshot from `OpenVINO
documentation <https://docs.openvino.ai/2023.1/get_started.html#openvino-advanced-features>`__

.. code:: ipython3

    import requests
    from PIL import Image
    from io import BytesIO
    
    
    def load_image(image_file):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    
    test_image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/aa46ef0c-c14d-4bab-8bb7-3b22fe73f6bc"
    
    image = load_image(test_image_url)
    text = "What performance hints do?"
    
    inputs = processor(images=image, text=text, return_tensors="pt")
    display(image)



.. image:: 260-pix2struct-docvqa-with-output_files/260-pix2struct-docvqa-with-output_11_0.png


.. code:: ipython3

    answer_tokens = ov_model.generate(**inputs)
    answer = processor.decode(answer_tokens[0], skip_special_tokens=True)
    print(f"Question: {text}")
    print(f"Answer: {answer}")


.. parsed-literal::

    /home/ea/work/ov_venv/lib/python3.8/site-packages/optimum/intel/openvino/modeling_seq2seq.py:395: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      last_hidden_state = torch.from_numpy(self.request(inputs, shared_memory=True)["last_hidden_state"]).to(
    /home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/generation/utils.py:1260: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
      warnings.warn(
    /home/ea/work/ov_venv/lib/python3.8/site-packages/optimum/intel/openvino/modeling_seq2seq.py:476: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      self.request.start_async(inputs, shared_memory=True)


.. parsed-literal::

    Question: What performance hints do?
    Answer: automatically adjust runtime parameters to prioritize for low latency or high throughput


Interactive demo
----------------



.. code:: ipython3

    import gradio as gr
    
    example_images_urls = [
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/94ef687c-aebb-452b-93fe-c7f29ce19503",
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/70b2271c-9295-493b-8a5c-2f2027dcb653",
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1e2be134-0d45-4878-8e6c-08cfc9c8ea3d"
    ]
    
    file_names = ["eiffel_tower.png", "exsibition.jpeg", "population_table.jpeg"]
    
    for img_url, image_file in zip(example_images_urls, file_names):
        load_image(img_url).save(image_file)
    
    questions = ["What is Eiffel tower tall?", "When is the coffee break?", "What the population of Stoddard?"] 
    
    examples = [list(pair) for pair in zip(file_names, questions)]
    
    def generate(img, question):
        inputs = processor(images=img, text=question, return_tensors="pt")
        predictions = ov_model.generate(**inputs, max_new_tokens=256)
        return processor.decode(predictions[0], skip_special_tokens=True)
    
    demo = gr.Interface(
        fn=generate,
        inputs=["image", "text"],
        outputs="text",
        title="Pix2Struct for DocVQA",
        examples=examples,
        cache_examples=False,
        allow_flagging="never",
    )
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
