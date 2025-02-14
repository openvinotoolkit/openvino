Multimodal RAG for video analytics with LlamaIndex
==================================================

Constructing a RAG pipeline for text is relatively straightforward,
thanks to the tools developed for parsing, indexing, and retrieving text
data. However, adapting RAG models for video content presents a greater
challenge. Videos combine visual, auditory, and textual elements,
requiring more processing power and sophisticated video pipelines.

To build a truly multimodal search for videos, you need to work with
different modalities of a video like spoken content, visual. In this
notebook, we showcase a Multimodal RAG pipeline designed for video
analytics. It utilizes Whisper model to convert spoken content to text,
CLIP model to generate multimodal embeddings, and Vision Language model
(VLM) to process retrieved images and text messages. The following
picture illustrates how this pipeline is working.

.. figure:: https://github.com/user-attachments/assets/baef4914-5c07-432c-9363-1a0cb5944b09
   :alt: Multimodal RAG

   Multimodal RAG


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert and Compress models <#convert-and-compress-models>`__

   -  `ASR model <#asr-model>`__
   -  `CLIP model <#clip-model>`__
   -  `VLM model <#vlm-model>`__

-  `Download and process video <#download-and-process-video>`__

   -  `Initialize ASR <#initialize-asr>`__

-  `Create the multi-modal index <#create-the-multi-modal-index>`__

   -  `Initialize CLIP <#initialize-clip>`__

-  `Search text and image
   embeddings <#search-text-and-image-embeddings>`__
-  `Generate final response using
   VLM <#generate-final-response-using-vlm>`__

   -  `Set the RAG prompt template <#set-the-rag-prompt-template>`__
   -  `Initialize VLM <#initialize-vlm>`__

-  `Interactive Demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



install required packages and setup helper functions.

.. code:: ipython3

    %pip uninstall -q -y "moviepy" "decorator"
    
    %pip install -q "llama-index-core" "llama-index-embeddings-openvino>=0.4.1" "llama-index-multi-modal-llms-openvino" "llama-index-readers-file" \
        "llama-index-vector-stores-qdrant"  \
        "transformers>=4.45" \
        "moviepy>=2.1.1" \
        "librosa" \
        "python-ffmpeg<=1.0.16" \
        "open_clip_torch" \
        "huggingface_hub" \
        "gradio>=4.44.1" --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    

.. code:: ipython3

    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" "openvino>=2024.5.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    

.. code:: ipython3

    import os
    import requests
    from pathlib import Path
    
    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w", encoding="utf-8").write(r.text)
    
    if not Path("cmd_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py")
        open("cmd_helper.py", "w", encoding="utf-8").write(r.text)
    
    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry
    
    collect_telemetry("multimodal-rag-llamaindex.ipynb")

Convert and Compress models
---------------------------



ASR model
~~~~~~~~~



In this example, we utilize
`Distil-Whisper <https://huggingface.co/distil-whisper/distil-large-v2>`__
to recognize the spoken content in video and generate text.
Distil-Whisper is a distilled variant of the
`Whisper <https://huggingface.co/openai/whisper-large-v2>`__ model by
OpenAI. The Distil-Whisper is proposed in the paper `Robust Knowledge
Distillation via Large-Scale Pseudo
Labelling <https://arxiv.org/abs/2311.00430>`__. According to authors,
compared to Whisper, Distil-Whisper runs in several times faster with
50% fewer parameters, while performing to within 1% word error rate
(WER) on out-of-distribution evaluation data. For more information about
Distil-Whisper, please refer `Distil-Whisper
notebook <distil-whisper-asr-with-output.html>`__.

.. code:: ipython3

    import huggingface_hub as hf_hub
    
    asr_model_id = "OpenVINO/distil-whisper-large-v3-int8-ov"
    asr_model_path = asr_model_id.split("/")[-1]
    
    if not Path(asr_model_path).exists():
        hf_hub.snapshot_download(asr_model_id, local_dir=asr_model_path)

CLIP model
~~~~~~~~~~



In this example, CLIP model will help to generate the embedding vectors
for both text and images. CLIP (Contrastive Language-Image Pre-Training)
is a neural network trained on various (image, text) pairs. It can be
instructed in natural language to predict the most relevant text
snippet, given an image, without directly optimizing for the task.

CLIP uses a `ViT <https://arxiv.org/abs/2010.11929>`__ like transformer
to get visual features and a causal language model to get the text
features. The text and visual features are then projected into a latent
space with identical dimensions. The dot product between the projected
image and text features is then used as a similarity score.

.. code:: ipython3

    from cmd_helper import optimum_cli
    
    clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    clip_model_path = clip_model_id.split("/")[-1]
    
    if not Path(clip_model_path).exists():
        optimum_cli(clip_model_id, clip_model_path)



**Export command:**



``optimum-cli export openvino --model laion/CLIP-ViT-B-32-laion2B-s34B-b79K CLIP-ViT-B-32-laion2B-s34B-b79K``


VLM model
~~~~~~~~~



Vision Language model (VLM) is used to generate final response regrading
the context of images and texts retrieved from vector DB. It can help to
understand the both language and image instructions to complete various
real-world tasks. In this example, we select
`Phi-3.5-Vision <https://huggingface.co/microsoft/Phi-3.5-vision-instruct>`__
as VLM.

The Phi-3-Vision is a lightweight, state-of-the-art open multimodal
model built upon datasets which include - synthetic data and filtered
publicly available websites - with a focus on very high-quality,
reasoning dense data both on text and vision. The model belongs to the
Phi-3 model family, and the multimodal version comes with 128K context
length (in tokens) it can support. The model underwent a rigorous
enhancement process, incorporating both supervised fine-tuning and
direct preference optimization to ensure precise instruction adherence
and robust safety measures. More details about model can be found in
`model blog
post <https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/>`__,
`technical report <https://aka.ms/phi3-tech-report>`__,
`Phi-3-cookbook <https://github.com/microsoft/Phi-3CookBook>`__

.. code:: ipython3

    vlm_model_id = "microsoft/Phi-3.5-vision-instruct"
    vlm_model_path = Path(vlm_model_id.split("/")[-1]) / "FP16"
    
    if not vlm_model_path.exists():
        !optimum-cli export openvino --model {vlm_model_id} --weight-format fp16 {vlm_model_path} --trust-remote-code

.. code:: ipython3

    import shutil
    import nncf
    import openvino as ov
    import gc
    
    core = ov.Core()
    
    compression_config = {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 64,
        "ratio": 0.6,
    }
    
    compressed_model_path = vlm_model_path.parent / "INT4"
    if not compressed_model_path.exists():
        ov_model = core.read_model(vlm_model_path / "openvino_language_model.xml")
        compressed_ov_model = nncf.compress_weights(ov_model, **compression_config)
        ov.save_model(
            compressed_ov_model,
            compressed_model_path / "openvino_language_model.xml",
        )
        del compressed_ov_model
        del ov_model
        gc.collect()
        for file_name in vlm_model_path.glob("*"):
            if file_name.name in [
                "openvino_language_model.xml",
                "openvino_language_model.bin",
            ]:
                continue
            shutil.copy(file_name, compressed_model_path)

Download and process video
--------------------------



To begin, download an example video from YouTube and extract the audio
and frame files from it.

.. code:: ipython3

    video_url = "https://github.com/user-attachments/assets/2e38afa5-ce3e-448c-ad1a-0e9471039ff1"
    output_folder = "./mixed_data/"
    output_audio_path = "./mixed_data/output_audio.wav"
    filepath = "video_data/input_vid.mp4"
    
    example_path = Path(filepath)
    example_path.parent.mkdir(parents=True, exist_ok=True)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    if not example_path.exists():
        r = requests.get(video_url)
        with example_path.open("wb") as f:
            f.write(r.content)

Initialize ASR
~~~~~~~~~~~~~~



Select inference device

.. code:: ipython3

    from notebook_utils import device_widget
    
    asr_device = device_widget(default="AUTO", exclude=["NPU"])
    
    asr_device




.. parsed-literal::

    Dropdown(description='Device:', index=2, options=('CPU', 'GPU', 'AUTO'), value='AUTO')



The Hugging Face Optimum API is a high-level API that enables us to
convert and quantize models from the Hugging Face Transformers library
to the OpenVINO™ IR format. For more details, refer to the `Hugging Face
Optimum
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.

Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. The Optimum Inference models are API compatible with Hugging
Face Transformers models. This means we just need to replace the
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

.. code:: ipython3

    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline
    
    asr_model = OVModelForSpeechSeq2Seq.from_pretrained(asr_model_path, device=asr_device.value)
    asr_processor = AutoProcessor.from_pretrained(asr_model_path)
    
    pipe = pipeline("automatic-speech-recognition", model=asr_model, tokenizer=asr_processor.tokenizer, feature_extractor=asr_processor.feature_extractor)

.. code:: ipython3

    import librosa
    from moviepy.video.io.VideoFileClip import VideoFileClip
    
    
    def video_to_images(video_path, output_folder):
        """
        Convert a video to a sequence of images and save them to the output folder.
    
        Params:
        video_path (str): The path to the video file.
        output_folder (str): The path to the folder to save the images to.
    
        """
        clip = VideoFileClip(video_path)
        clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.1)
    
    
    def video_to_audio(video_path, output_audio_path):
        """
        Convert a video to audio and save it to the output path.
    
        Params:
        video_path (str): The path to the video file.
        output_audio_path (str): The path to save the audio to.
    
        """
        clip = VideoFileClip(video_path)
        audio = clip.audio
        audio.write_audiofile(output_audio_path)
    
    
    def audio_to_text(audio_path):
        """
        Convert audio to text using the SpeechRecognition library.
    
        Params:
        audio_path (str): The path to the audio file.
    
        Returns:
        test (str): The text recognized from the audio.
    
        """
        en_raw_speech, samplerate = librosa.load(audio_path, sr=16000)
        result = pipe(en_raw_speech, return_timestamps=True)
    
        return result["text"]

In this step, we will extract the images and audio from video, then
convert its audio into text.

.. code:: ipython3

    try:
        video_to_images(filepath, output_folder)
        video_to_audio(filepath, output_audio_path)
        text_data = audio_to_text(output_audio_path)
    
        with open(output_folder + "output_text.txt", "w") as file:
            file.write(text_data)
        print("Text data saved to file")
        file.close()
        os.remove(output_audio_path)
    
    except Exception as e:
        raise e


.. parsed-literal::

    Moviepy - Writing frames ./mixed_data/frame%04d.png.
    



                                                                                                                                                                                                                                                                                                                                                     

.. parsed-literal::

    Moviepy - Done writing frames ./mixed_data/frame%04d.png.
    MoviePy - Writing audio in ./mixed_data/output_audio.wav
    



                                                                                                                                                                                                                                                                                                                                                     

.. parsed-literal::

    MoviePy - Done.
    

.. parsed-literal::

    You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50360]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
    

.. parsed-literal::

    Text data saved to file
    

Create the multi-modal index
----------------------------



In this step, we are going to build multi-modal index and vector store
to index both text and images. The CLIP model is used to generate the
embedding vector for texts and images.

Initialize CLIP
~~~~~~~~~~~~~~~



Select inference device

.. code:: ipython3

    clip_device = device_widget(default="AUTO", exclude=["NPU"])
    
    clip_device




.. parsed-literal::

    Dropdown(description='Device:', index=2, options=('CPU', 'GPU', 'AUTO'), value='AUTO')



Class ``OpenVINOClipEmbedding`` in LlamaIndex can support exporting and
loading open_clip models with OpenVINO runtime. for more information,
please refer `Local Embeddings with
OpenVINO <https://docs.llamaindex.ai/en/stable/examples/embeddings/openvino/#openclip-model-exporter>`__.

.. code:: ipython3

    from llama_index.embeddings.huggingface_openvino import OpenVINOClipEmbedding
    
    clip_model = OpenVINOClipEmbedding(model_id_or_path=clip_model_path, device=clip_device.value)

.. code:: ipython3

    import torch
    
    if hasattr(torch, "mps") and torch.mps.is_available:
        torch.mps.is_available = lambda: False
    
    from llama_index.core.indices import MultiModalVectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.core import StorageContext, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import SimpleDirectoryReader
    import qdrant_client
    
    # Create the MultiModal index
    documents = SimpleDirectoryReader(output_folder).load_data()
    
    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(":memory:")
    
    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")
    storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

.. code:: ipython3

    Settings.embed_model = clip_model
    
    index = MultiModalVectorStoreIndex.from_documents(
        documents, storage_context=storage_context, image_embed_model=Settings.embed_model, transformations=[SentenceSplitter(chunk_size=300, chunk_overlap=30)]
    )
    
    retriever_engine = index.as_retriever(similarity_top_k=2, image_similarity_top_k=5)


.. parsed-literal::

    WARNING:root:Payload indexes have no effect in the local Qdrant. Please use server Qdrant if you need payload indexes.
    WARNING:root:Payload indexes have no effect in the local Qdrant. Please use server Qdrant if you need payload indexes.
    

Search text and image embeddings
--------------------------------



To simply the prompt for VLM, we have to prepare the context of text and
images regarding user’s query. In this step, the most relevant context
will be retrieved from vector DB through multi-modal index.

.. code:: ipython3

    from llama_index.core.response.notebook_utils import display_source_node
    from llama_index.core.schema import ImageNode
    from PIL import Image
    import matplotlib.pyplot as plt
    import os
    
    
    def plot_images(image_paths):
        images_shown = 0
        plt.figure(figsize=(16, 9))
        for img_path in image_paths:
            if os.path.isfile(img_path):
                image = Image.open(img_path)
    
                plt.subplot(2, 3, images_shown + 1)
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])
    
                images_shown += 1
                if images_shown >= 7:
                    break
    
    
    def retrieve(retriever_engine, query_str):
        retrieval_results = retriever_engine.retrieve(query_str)
    
        retrieved_image = []
        retrieved_text = []
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                retrieved_image.append(res_node.node.metadata["file_path"])
            else:
                display_source_node(res_node, source_length=200)
                retrieved_text.append(res_node.text)
    
        return retrieved_image, retrieved_text

.. code:: ipython3

    query_str = "tell me more about gaussian function"
    
    img, txt = retrieve(retriever_engine=retriever_engine, query_str=query_str)
    image_documents = SimpleDirectoryReader(input_dir=output_folder, input_files=img).load_data()
    context_str = "".join(txt)
    plot_images(img)
    print(txt)



**Node ID:** 72d9a990-ba98-4281-be50-42ef56254cad\ **Similarity:**
0.6902350328083398\ **Text:** The basic function underlying a normal
distribution, aka a Gaussian, is E to the negative x squared. But you
might wonder why this function? Of all the expressions we could dream up
that give you s…



**Node ID:** 5989233f-c4dd-4394-802f-5b7d0fa1cacc\ **Similarity:**
0.679477746185225\ **Text:** I’d like to share an especially pleasing
visual way that you can think about this calculation, which hopefully
offers some sense of what makes the E to the negative x squared function
special in th…


.. parsed-literal::

    ["The basic function underlying a normal distribution, aka a Gaussian, is E to the negative x squared. But you might wonder why this function? Of all the expressions we could dream up that give you some symmetric smooth graph with mass concentrated towards the middle, why is it that the theory of probability seems to have a special place in its heart for this particular expression? For the last many videos I've been hinting at an answer to this question, and here we'll finally arrive at something like a satisfying answer. As a quick refresher on where we are, a couple videos ago we talked about the central limit theorem, which describes how as you add multiple copies of a random variable, for example rolling a weighted die many different times or letting a ball bounce off of a peg repeatedly, then the distribution describing that sum tends to look approximately like a normal distribution. What the central limit theorem says is as you make that sum bigger and bigger, under appropriate conditions, that approximation to a normal becomes better and better. But I never explained why this theorem is actually true. We only talked about what it's claiming. In the last video, we started talking about the math involved in adding two random variables. If you have two random variables each following some distribution, then to find the distribution describing the sum of those variables, you compute something known as a convolution between the two original functions. And we spent a lot of time building up two distinct ways to visualize what this convolution operation really is. really is. Today, our basic job is to work through a particular example, which is to ask, what happens when you add two normally distributed random variables, which, as you know by now, is the same as asking what do you get if you compute a convolution between two Gaussian functions?", "I'd like to share an especially pleasing visual way that you can think about this calculation, which hopefully offers some sense of what makes the E to the negative x squared function special in the first place. After we walk through it, we'll talk about how this calculation is one of the steps involved improving the central limit theorem. It's the step that answers the question of why a Gaussian and not something else is the central limit. But first, let's dive in. The full formula for a Gaussian is more complicated than just e to the negative x squared. The exponent is typically written as negative 1 half times x divided by sigma squared, where sigma describes the spread of the distribution, specifically the standard deviation. All of this needs to be multiplied by a fraction on the front, which is there to make sure that the area under the curve is one, making it a valid probability distribution, and if you want to consider distributions that aren't necessarily centered at zero, you would also throw another parameter mu into the exponent like this, although for everything we'll be doing here we just consider centered distributions. Now if you look at our central goal for today, which is to compute a convolution between two Gaussian functions, The direct way to do this would be to take the definition of a convolution, this integral expression we built up last video, and then to plug in for each one of the functions involved, the formula for a Gaussian. It's kind of a lot of symbols when you throw it all together, but more than anything, working this out, is an exercise in completing the square. And there's nothing wrong with that. That will get you the answer that you want."]
    


.. image:: multimodal-rag-llamaindex-with-output_files/multimodal-rag-llamaindex-with-output_29_3.png


Generate final response using VLM
---------------------------------



Set the RAG prompt template
~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    qa_tmpl_str = (
        "Given the provided information, including relevant images and retrieved context from the video, \
     accurately and precisely answer the query without any additional prior knowledge.\n"
        "Please ensure honesty and responsibility, refraining from any racist or sexist remarks.\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "---------------------\n"
        "Query: {query_str}\n"
        "Answer: "
    )

Initialize VLM
~~~~~~~~~~~~~~



Select inference device

.. code:: ipython3

    vlm_device = device_widget(default="AUTO", exclude=["NPU"])
    
    vlm_device




.. parsed-literal::

    Dropdown(description='Device:', index=2, options=('CPU', 'GPU', 'AUTO'), value='AUTO')



``OpenVINOMultiModal`` class provides convenient way for running
multimodal model in LlamaIndex. It accepts directory with converted
model and inference device as arguments. For running model with
streaming we will use ``stream_complete`` method. For more information
about the OpenVINO multimodal models support in LlamaIndex, refer to the
`OpenVINOMultiModal
Document <https://docs.llamaindex.ai/en/stable/examples/multi_modal/openvino_multimodal/>`__.

.. code:: ipython3

    from transformers import AutoProcessor, AutoTokenizer
    
    vlm_int4_model_path = "Phi-3.5-vision-instruct/INT4"
    
    processor = AutoProcessor.from_pretrained(vlm_int4_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(vlm_int4_model_path)
    
    
    def messages_to_prompt(messages, image_documents):
        """
        Prepares the input messages and images.
        """
        images = []
        placeholder = ""
    
        for i, img_doc in enumerate(image_documents, start=1):
            images.append(Image.open(img_doc.image_path))
            placeholder += f"<|image_{i}|>\n"
        conversation = [
            {"role": "user", "content": placeholder + messages[0].content},
        ]
    
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
        inputs = processor(prompt, images, return_tensors="pt")
        return inputs

.. code:: ipython3

    from llama_index.multi_modal_llms.openvino import OpenVINOMultiModal
    
    vlm = OpenVINOMultiModal(
        model_id_or_path=vlm_int4_model_path,
        device=vlm_device.value,
        messages_to_prompt=messages_to_prompt,
        trust_remote_code=True,
        generate_kwargs={"do_sample": False, "eos_token_id": processor.tokenizer.eos_token_id},
    )

.. code:: ipython3

    response = vlm.stream_complete(
        prompt=qa_tmpl_str.format(context_str=context_str, query_str=query_str),
        image_documents=image_documents,
    )
    for r in response:
        print(r.delta, end="")


.. parsed-literal::

    A Gaussian function, also known as a normal distribution, is a type of probability distribution that is symmetric and bell-shaped. It is characterized by its mean and standard deviation, which determine the center and spread of the distribution, respectively. The Gaussian function is widely used in statistics and probability theory due to its unique properties and applications in various fields such as physics, engineering, and finance. The function is defined by the equation e to the negative x squared, where x represents the input variable. The graph of a Gaussian function is a smooth curve that approaches the x-axis as it moves away from the center, creating a bell-like shape. The function is also known for its property of being able to describe the distribution of random variables, making it a fundamental concept in probability theory and statistics.

Interactive Demo
----------------



Now, you can try to chat with model. Upload video, provide your text
message into ``Input`` field and click ``Submit`` to start
communication.

.. code:: ipython3

    import gradio as gr
    import base64
    
    
    def path2base64(path):
        with open(path, "rb") as f:
            byte_data = f.read()
        base64_str = base64.b64encode(byte_data).decode("ascii")
        return base64_str
    
    
    def build_index(video_path):
        """
        callback function for building index of vector store
    
        Params:
          video_path: path of uploaded video file
        Returns:
          vector store is ready
    
        """
    
        global retriever_engine
        progress = gr.Progress()
        progress(None, desc="Video to Images...")
        video_to_images(video_path, output_folder)
        progress(None, desc="Video to Audio...")
        video_to_audio(video_path, output_audio_path)
        progress(None, desc="Audio to Texts...")
        text_data = audio_to_text(output_audio_path)
    
        with open(output_folder + "output_text.txt", "w") as file:
            file.write(text_data)
        print("Text data saved to file")
        file.close()
        os.remove(output_audio_path)
    
        progress(0, desc="Building Index...")
        documents = SimpleDirectoryReader(output_folder).load_data()
        client = qdrant_client.QdrantClient(":memory:")
    
        text_store = QdrantVectorStore(client=client, collection_name="text_collection")
        image_store = QdrantVectorStore(client=client, collection_name="image_collection")
        storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
        index = MultiModalVectorStoreIndex.from_documents(
            documents, storage_context=storage_context, image_embed_model=Settings.embed_model, transformations=[SentenceSplitter(chunk_size=300, chunk_overlap=30)]
        )
    
        retriever_engine = index.as_retriever(similarity_top_k=2, image_similarity_top_k=5)
        return "Vector Store is Ready"
    
    
    def search(history):
        """
        callback function for searching vector store
    
        Params:
          history: conversation history
        Returns:
          lists of retrieved images and texts
    
        """
        progress = gr.Progress()
        progress(None, desc="Searching...")
        img, txt = retrieve(retriever_engine=retriever_engine, query_str=history[-1][0])
        return img, txt
    
    
    def generate(history, images, texts):
        """
        callback function for running chatbot on submit button click
    
        Params:
          history: conversation history
          images: list of retrieved images
          texts: list of retrieved texts
    
        """
        progress = gr.Progress()
        progress(None, desc="Generating...")
        image_documents = SimpleDirectoryReader(input_dir=output_folder, input_files=images).load_data()
    
        context_str = "".join(texts)
    
        response = vlm.stream_complete(
            prompt=qa_tmpl_str.format(context_str=context_str, query_str=history[-1][0]),
            image_documents=image_documents,
        )
        images_list = ""
        for image in images:
            image_base64 = path2base64(image)
            images_list += f'<img src="data:image/png;base64,{image_base64}">'
        images_list += "\n"
        partial_text = "According to audio and following screenshots from the video: \n"
        partial_text += images_list
        for r in response:
            partial_text += r.delta
            history[-1][1] = partial_text
            yield history
    
    
    def stop():
        vlm._model.request.cancel()

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/multimodal-rag/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(filepath, build_index, search, generate, stop)
    
    try:
        demo.queue().launch()
    except Exception:
        demo.queue().launch(share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
