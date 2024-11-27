Visual Content Search using MobileCLIP and OpenVINO
===================================================

Semantic visual content search is a machine learning task that uses
either a text query or an input image to search a database of images
(photo gallery, video) to find images that are semantically similar to
the search query. Historically, building a robust search engine for
images was difficult. One could search by features such as file name and
image metadata, and use any context around an image (i.e. alt text or
surrounding text if an image appears in a passage of text) to provide
the richer searching feature. This was before the advent of neural
networks that can identify semantically related images to a given user
query.

`Contrastive Language-Image Pre-Training
(CLIP) <https://arxiv.org/abs/2103.00020>`__ models provide the means
through which you can implement a semantic search engine with a few
dozen lines of code. The CLIP model has been trained on millions of
pairs of text and images, encoding semantics from images and text
combined. Using CLIP, you can provide a text query and CLIP will return
the images most related to the query.

In this tutorial, we consider how to use MobileCLIP to implement a
visual content search engine for finding relevant frames in video.

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Select model <#select-model>`__
-  `Run model inference <#run-model-inference>`__

   -  `Prepare image gallery <#prepare-image-gallery>`__
   -  `Prepare model <#prepare-model>`__
   -  `Perform search <#perform-search>`__

-  `Convert Model to OpenVINO Intermediate Representation
   format <#convert-model-to-openvino-intermediate-representation-format>`__
-  `Run OpenVINO model inference <#run-openvino-model-inference>`__

   -  `Select device for image
      encoder <#select-device-for-image-encoder>`__
   -  `Select device for text
      encoder <#select-device-for-text-encoder>`__
   -  `Perform search <#perform-search>`__

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



.. code:: ipython3

    import requests


    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py",
    )
    open("cmd_helper.py", "w").write(r.text)




.. parsed-literal::

    1491



.. code:: ipython3

    from cmd_helper import clone_repo


    clone_repo("https://github.com/apple/ml-mobileclip.git")




.. parsed-literal::

    PosixPath('ml-mobileclip')



.. code:: ipython3

    %pip install -q "./ml-mobileclip" --no-deps

    %pip install -q "clip-benchmark>=1.4.0" "datasets>=2.8.0" "open-clip-torch>=2.20.0" "timm>=0.9.5" "torch>=2.5.0" "torchvision>=0.20.0" --extra-index-url https://download.pytorch.org/whl/cpu

    %pip install -q "matplotlib>=3.4" "Pillow"  "altair" "pandas" "tqdm" "salesforce-lavis==1.0.2"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    ERROR: Could not find a version that satisfies the requirement torch>=2.5.0 (from versions: 1.4.0, 1.4.0+cpu, 1.5.0, 1.5.0+cpu, 1.5.1, 1.5.1+cpu, 1.6.0, 1.6.0+cpu, 1.7.0, 1.7.0+cpu, 1.7.1, 1.7.1+cpu, 1.8.0, 1.8.0+cpu, 1.8.1, 1.8.1+cpu, 1.9.0, 1.9.0+cpu, 1.9.1, 1.9.1+cpu, 1.10.0, 1.10.0+cpu, 1.10.1, 1.10.1+cpu, 1.10.2, 1.10.2+cpu, 1.11.0, 1.11.0+cpu, 1.12.0, 1.12.0+cpu, 1.12.1, 1.12.1+cpu, 1.13.0, 1.13.0+cpu, 1.13.1, 1.13.1+cpu, 2.0.0, 2.0.0+cpu, 2.0.1, 2.0.1+cpu, 2.1.0, 2.1.0+cpu, 2.1.1, 2.1.1+cpu, 2.1.2, 2.1.2+cpu, 2.2.0, 2.2.0+cpu, 2.2.1, 2.2.1+cpu, 2.2.2, 2.2.2+cpu, 2.3.0, 2.3.0+cpu, 2.3.1, 2.3.1+cpu, 2.4.0, 2.4.0+cpu, 2.4.1, 2.4.1+cpu)
    ERROR: No matching distribution found for torch>=2.5.0
    Note: you may need to restart the kernel to use updated packages.
      error: subprocess-exited-with-error

      × pip subprocess to install build dependencies did not run successfully.
      │ exit code: 1
      ╰─> [68 lines of output]
          Ignoring numpy: markers 'python_version >= "3.9"' don't match your environment
          Collecting setuptools
            Using cached setuptools-75.3.0-py3-none-any.whl.metadata (6.9 kB)
          Collecting cython<3.0,>=0.25
            Using cached Cython-0.29.37-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl.metadata (3.1 kB)
          Collecting cymem<2.1.0,>=2.0.2
            Using cached cymem-2.0.8-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.4 kB)
          Collecting preshed<3.1.0,>=3.0.2
            Using cached preshed-3.0.9-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.2 kB)
          Collecting murmurhash<1.1.0,>=0.28.0
            Using cached murmurhash-1.0.10-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.0 kB)
          Collecting thinc<8.4.0,>=8.3.0
            Using cached thinc-8.3.2.tar.gz (193 kB)
            Installing build dependencies: started
            Installing build dependencies: finished with status 'error'
            error: subprocess-exited-with-error

            × pip subprocess to install build dependencies did not run successfully.
            │ exit code: 1
            ╰─> [38 lines of output]
                Ignoring numpy: markers 'python_version >= "3.9"' don't match your environment
                Collecting setuptools
                  Using cached setuptools-75.3.0-py3-none-any.whl.metadata (6.9 kB)
                Collecting cython<3.0,>=0.25
                  Using cached Cython-0.29.37-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl.metadata (3.1 kB)
                Collecting murmurhash<1.1.0,>=1.0.2
                  Using cached murmurhash-1.0.10-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.0 kB)
                Collecting cymem<2.1.0,>=2.0.2
                  Using cached cymem-2.0.8-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.4 kB)
                Collecting preshed<3.1.0,>=3.0.2
                  Using cached preshed-3.0.9-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.2 kB)
                Collecting blis<1.1.0,>=1.0.0
                  Using cached blis-1.0.1.tar.gz (3.6 MB)
                  Installing build dependencies: started
                  Installing build dependencies: finished with status 'error'
                  error: subprocess-exited-with-error

                  × pip subprocess to install build dependencies did not run successfully.
                  │ exit code: 1
                  ╰─> [8 lines of output]
                      Collecting setuptools
                        Using cached setuptools-75.3.0-py3-none-any.whl.metadata (6.9 kB)
                      Collecting cython>=0.25
                        Using cached Cython-3.0.11-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.2 kB)
                      ERROR: Ignored the following versions that require a different python version: 1.25.0 Requires-Python >=3.9; 1.25.1 Requires-Python >=3.9; 1.25.2 Requires-Python >=3.9; 1.26.0 Requires-Python <3.13,>=3.9; 1.26.1 Requires-Python <3.13,>=3.9; 1.26.2 Requires-Python >=3.9; 1.26.3 Requires-Python >=3.9; 1.26.4 Requires-Python >=3.9; 2.0.0 Requires-Python >=3.9; 2.0.1 Requires-Python >=3.9; 2.0.2 Requires-Python >=3.9; 2.1.0 Requires-Python >=3.10; 2.1.0rc1 Requires-Python >=3.10; 2.1.1 Requires-Python >=3.10; 2.1.2 Requires-Python >=3.10; 2.1.3 Requires-Python >=3.10; 75.4.0 Requires-Python >=3.9; 75.5.0 Requires-Python >=3.9; 75.6.0 Requires-Python >=3.9
                      ERROR: Could not find a version that satisfies the requirement numpy<3.0.0,>=2.0.0 (from versions: 1.3.0, 1.4.1, 1.5.0, 1.5.1, 1.6.0, 1.6.1, 1.6.2, 1.7.0, 1.7.1, 1.7.2, 1.8.0, 1.8.1, 1.8.2, 1.9.0, 1.9.1, 1.9.2, 1.9.3, 1.10.0.post2, 1.10.1, 1.10.2, 1.10.4, 1.11.0, 1.11.1, 1.11.2, 1.11.3, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 1.13.3, 1.14.0, 1.14.1, 1.14.2, 1.14.3, 1.14.4, 1.14.5, 1.14.6, 1.15.0, 1.15.1, 1.15.2, 1.15.3, 1.15.4, 1.16.0, 1.16.1, 1.16.2, 1.16.3, 1.16.4, 1.16.5, 1.16.6, 1.17.0, 1.17.1, 1.17.2, 1.17.3, 1.17.4, 1.17.5, 1.18.0, 1.18.1, 1.18.2, 1.18.3, 1.18.4, 1.18.5, 1.19.0, 1.19.1, 1.19.2, 1.19.3, 1.19.4, 1.19.5, 1.20.0, 1.20.1, 1.20.2, 1.20.3, 1.21.0, 1.21.1, 1.21.2, 1.21.3, 1.21.4, 1.21.5, 1.21.6, 1.22.0, 1.22.1, 1.22.2, 1.22.3, 1.22.4, 1.23.0, 1.23.1, 1.23.2, 1.23.3, 1.23.4, 1.23.5, 1.24.0, 1.24.1, 1.24.2, 1.24.3, 1.24.4)
                      ERROR: No matching distribution found for numpy<3.0.0,>=2.0.0

                      [end of output]

                  note: This error originates from a subprocess, and is likely not a problem with pip.
                error: subprocess-exited-with-error

                × pip subprocess to install build dependencies did not run successfully.
                │ exit code: 1
                ╰─> See above for output.

                note: This error originates from a subprocess, and is likely not a problem with pip.
                [end of output]

            note: This error originates from a subprocess, and is likely not a problem with pip.
          error: subprocess-exited-with-error

          × pip subprocess to install build dependencies did not run successfully.
          │ exit code: 1
          ╰─> See above for output.

          note: This error originates from a subprocess, and is likely not a problem with pip.
          [end of output]

      note: This error originates from a subprocess, and is likely not a problem with pip.
    error: subprocess-exited-with-error

    × pip subprocess to install build dependencies did not run successfully.
    │ exit code: 1
    ╰─> See above for output.

    note: This error originates from a subprocess, and is likely not a problem with pip.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" "openvino>=2024.0.0" "altair" "opencv-python" "opencv-contrib-python" "gradio>=4.19"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Select model
------------



For starting work, we should select model that will be used in our
demonstration. By default, we will use the MobileCLIP model, but for
comparison purposes, you can select different models among:

-  **CLIP** - CLIP (Contrastive Language-Image Pre-Training) is a neural
   network trained on various (image, text) pairs. It can be instructed
   in natural language to predict the most relevant text snippet, given
   an image, without directly optimizing for the task. CLIP uses a
   `ViT <https://arxiv.org/abs/2010.11929>`__ like transformer to get
   visual features and a causal language model to get the text features.
   The text and visual features are then projected into a latent space
   with identical dimensions. The dot product between the projected
   image and text features is then used as a similarity score. You can
   find more information about this model in the `research
   paper <https://arxiv.org/abs/2103.00020>`__, `OpenAI
   blog <https://openai.com/blog/clip/>`__, `model
   card <https://github.com/openai/CLIP/blob/main/model-card.md>`__ and
   GitHub `repository <https://github.com/openai/CLIP>`__.
-  **SigLIP** - The SigLIP model was proposed in `Sigmoid Loss for
   Language Image Pre-Training <https://arxiv.org/abs/2303.15343>`__.
   SigLIP proposes to replace the loss function used in
   `CLIP <https://github.com/openai/CLIP>`__ (Contrastive Language–Image
   Pre-training) by a simple pairwise sigmoid loss. This results in
   better performance in terms of zero-shot classification accuracy on
   ImageNet. You can find more information about this model in the
   `research paper <https://arxiv.org/abs/2303.15343>`__ and `GitHub
   repository <https://github.com/google-research/big_vision>`__,
-  **MobileCLIP** - MobileCLIP – a new family of efficient image-text
   models optimized for runtime performance along with a novel and
   efficient training approach, namely multi-modal reinforced training.
   The smallest variant MobileCLIP-S0 obtains similar zero-shot
   performance as OpenAI’s CLIP ViT-b16 model while being several times
   faster and 2.8x smaller. More details about model can be found in
   `research paper <https://arxiv.org/pdf/2311.17049.pdf>`__ and `GitHub
   repository <https://github.com/apple/ml-mobileclip>`__.
-  **BLIP-2** - BLIP2 was introduced in the paper `BLIP-2: Bootstrapping
   Language-Image Pre-training with Frozen Image Encoders and Large
   Language Models <https://arxiv.org/abs/2301.12597>`__ by Li et
   al. and first released in this
   `repository <https://github.com/salesforce/LAVIS/tree/main/projects/blip2>`__.
   It is a generic and efficient pre-training strategy that easily
   harvests development of pretrained vision models and large language
   models (LLMs) for vision-language pretraining. BLIP-2 consists of 3
   models: a CLIP-like image encoder, a Querying Transformer (Q-Former)
   and a large language model.

.. code:: ipython3

    from pathlib import Path

    import ipywidgets as widgets


    model_dir = Path("checkpoints")


    def default_image_probs(image_features, text_features):
        image_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        return image_probs


    def blip2_image_probs(image_features, text_features):
        image_probs = image_features[:, 0, :] @ text_features[:, 0, :].t()
        return image_probs


    supported_models = {
        "MobileCLIP": {
            "mobileclip_s0": {
                "model_name": "mobileclip_s0",
                "pretrained": model_dir / "mobileclip_s0.pt",
                "url": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt",
                "image_size": 256,
                "image_probs": default_image_probs,
            },
            "mobileclip_s1": {
                "model_name": "mobileclip_s1",
                "pretrained": model_dir / "mobileclip_s1.pt",
                "url": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt",
                "image_size": 256,
                "image_probs": default_image_probs,
            },
            "mobileclip_s2": {
                "model_name": "mobileclip_s0",
                "pretrained": model_dir / "mobileclip_s2.pt",
                "url": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt",
                "image_size": 256,
                "image_probs": default_image_probs,
            },
            "mobileclip_b": {
                "model_name": "mobileclip_b",
                "pretrained": model_dir / "mobileclip_b.pt",
                "url": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt",
                "image_size": 224,
                "image_probs": default_image_probs,
            },
            "mobileclip_blt": {
                "model_name": "mobileclip_b",
                "pretrained": model_dir / "mobileclip_blt.pt",
                "url": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt",
                "image_size": 224,
                "image_probs": default_image_probs,
            },
        },
        "CLIP": {
            "clip-vit-b-32": {
                "model_name": "ViT-B-32",
                "pretrained": "laion2b_s34b_b79k",
                "image_size": 224,
                "image_probs": default_image_probs,
            },
            "clip-vit-b-16": {
                "model_name": "ViT-B-16",
                "pretrained": "openai",
                "image_size": 224,
                "image_probs": default_image_probs,
            },
            "clip-vit-l-14": {
                "model_name": "ViT-L-14",
                "pretrained": "datacomp_xl_s13b_b90k",
                "image_size": 224,
                "image_probs": default_image_probs,
            },
            "clip-vit-h-14": {
                "model_name": "ViT-H-14",
                "pretrained": "laion2b_s32b_b79k",
                "image_size": 224,
                "image_probs": default_image_probs,
            },
        },
        "SigLIP": {
            "siglip-vit-b-16": {
                "model_name": "ViT-B-16-SigLIP",
                "pretrained": "webli",
                "image_size": 224,
                "image_probs": default_image_probs,
            },
            "siglip-vit-l-16": {
                "model_name": "ViT-L-16-SigLIP-256",
                "pretrained": "webli",
                "image_size": 256,
                "image_probs": default_image_probs,
            },
        },
        "Blip2": {
            "blip2_feature_extractor": {
                "model_name": "blip2_feature_extractor",
                "pretrained": "pretrain_vitL",
                "image_size": 224,
                "image_probs": blip2_image_probs,
            },
        },
    }


    model_type = widgets.Dropdown(options=supported_models.keys(), default="MobileCLIP", description="Model type:")
    model_type




.. parsed-literal::

    Dropdown(description='Model type:', options=('MobileCLIP', 'CLIP', 'SigLIP', 'Blip2'), value='MobileCLIP')



.. code:: ipython3

    available_models = supported_models[model_type.value]

    model_checkpoint = widgets.Dropdown(
        options=available_models.keys(),
        default=list(available_models),
        description="Model:",
    )

    model_checkpoint




.. parsed-literal::

    Dropdown(description='Model:', options=('mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b', 'mo…



.. code:: ipython3

    from notebook_utils import download_file, device_widget

    model_config = available_models[model_checkpoint.value]

Run model inference
-------------------



Now, let’s see model in action. We will try to find image, where some
specific object is represented using embeddings. Embeddings are a
numeric representation of data such as text and images. The model
learned to encode semantics about the contents of images in embedding
format. This ability turns the model into a powerful for solving various
tasks including image-text retrieval. To reach our goal we should:

1. Calculate embeddings for all of the images in our dataset;
2. Calculate a text embedding for a user query (i.e. “black dog” or
   “car”);
3. Compare the text embedding to the image embeddings to find related
   embeddings.

The closer two embeddings are, the more similar the contents they
represent are.

Prepare image gallery
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from typing import List
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image


    def visualize_result(images: List, query: str = "", selected: List[int] = None):
        """
        Utility function for visualization classification results
        params:
          images (List[Image]) - list of images for visualization
          query (str) - title for visualization
          selected (List[int]) - list of selected image indices from images
        returns:
          matplotlib.Figure
        """
        figsize = (20, 5)
        fig, axs = plt.subplots(1, 4, figsize=figsize, sharex="all", sharey="all")
        fig.patch.set_facecolor("white")
        list_axes = list(axs.flat)
        if query:
            fig.suptitle(query, fontsize=20)
        for idx, a in enumerate(list_axes):
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            a.grid(False)
            a.imshow(images[idx])
            if selected is not None and idx not in selected:
                mask = np.ones_like(np.array(images[idx]))
                a.imshow(mask, "jet", interpolation="none", alpha=0.75)
        return fig


    images_urls = [
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/282ce53e-912d-41aa-ab48-2a001c022d74",
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/9bb40168-82b5-4b11-ada6-d8df104c736c",
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/0747b6db-12c3-4252-9a6a-057dcf8f3d4e",
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bricks.png",
    ]
    image_names = ["red_panda.png", "cat.png", "raccoon.png", "dog.png"]
    sample_path = Path("data")
    sample_path.mkdir(parents=True, exist_ok=True)

    images = []
    for image_name, image_url in zip(image_names, images_urls):
        image_path = sample_path / image_name
        if not image_path.exists():
            download_file(image_url, filename=image_name, directory=sample_path)
        images.append(Image.open(image_path).convert("RGB").resize((640, 420)))

    input_labels = ["cat"]
    text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

    visualize_result(images, "image gallery");



.. parsed-literal::

    data/red_panda.png:   0%|          | 0.00/50.6k [00:00<?, ?B/s]



.. parsed-literal::

    data/cat.png:   0%|          | 0.00/54.5k [00:00<?, ?B/s]



.. parsed-literal::

    data/raccoon.png:   0%|          | 0.00/106k [00:00<?, ?B/s]



.. parsed-literal::

    data/dog.png:   0%|          | 0.00/716k [00:00<?, ?B/s]



.. image:: mobileclip-video-search-with-output_files/mobileclip-video-search-with-output_12_4.png


Prepare model
~~~~~~~~~~~~~



The code bellow download model weights, create model class instance and
preprocessing utilities

.. code:: ipython3

    import torch


    class Blip2Model(torch.nn.Module):
        def __init__(self, ln_vision, visual_encoder, query_tokens, q_former, vision_proj, text_proj, tokenizer):
            super().__init__()
            self.ln_vision = ln_vision
            self.visual_encoder = visual_encoder
            self.query_tokens = query_tokens
            self.q_former = q_former
            self.vision_proj = vision_proj
            self.text_proj = text_proj
            self.tok = tokenizer

        def encode_image(self, image):
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long)
            query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)

            query_output = self.q_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = self.vision_proj(image_embeds)

            return image_features

        def encode_text(self, input_ids, attention_mask):
            text_output = self.q_former.bert(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            return text_features

        def tokenizer(self, text_descriptions):
            input_ids = self.tok(text_descriptions, return_tensors="pt", padding=True).input_ids
            attention_mask = self.tok(text_descriptions, return_tensors="pt", padding=True).attention_mask
            text = {"input_ids": input_ids, "attention_mask": attention_mask}
            return text

.. code:: ipython3

    import torch
    import time
    import mobileclip
    import open_clip

    # instantiate model
    model_name = model_config["model_name"]
    pretrained = model_config["pretrained"]

    if model_type.value == "MobileCLIP":
        model_dir.mkdir(exist_ok=True)
        model_url = model_config["url"]
        download_file(model_url, directory=model_dir)
        model, _, preprocess = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = mobileclip.get_tokenizer(model_name)
    elif model_type.value == "Blip2":
        from lavis.models import load_model_and_preprocess

        model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=pretrained, is_eval=True)
        model = Blip2Model(model.ln_vision, model.visual_encoder, model.query_tokens, model.Qformer, model.vision_proj, model.text_proj, model.tokenizer)
        preprocess = vis_processors["eval"]
        tokenizer = model.tokenizer
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
      warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)



.. parsed-literal::

    checkpoints/mobileclip_s0.pt:   0%|          | 0.00/206M [00:00<?, ?B/s]


Perform search
~~~~~~~~~~~~~~



.. code:: ipython3

    image_tensor = torch.stack([preprocess(image) for image in images])
    text = tokenizer(text_descriptions)
    image_probs_function = model_config["image_probs"]

    with torch.no_grad():
        # calculate image embeddings
        image_encoding_start = time.perf_counter()
        image_features = model.encode_image(image_tensor)
        image_encoding_end = time.perf_counter()
        print(f"Image encoding took {image_encoding_end - image_encoding_start:.3} ms")
        # calculate text embeddings
        text_encoding_start = time.perf_counter()
        text_features = model.encode_text(**text) if model_type.value == "Blip2" else model.encode_text(text)
        text_encoding_end = time.perf_counter()
        print(f"Text encoding took {text_encoding_end - text_encoding_start:.3} ms")

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_probs = image_probs_function(image_features, text_features)
        selected_image = [torch.argmax(image_probs).item()]

    visualize_result(images, input_labels[0], selected_image);


.. parsed-literal::

    Image encoding took 0.0979 ms
    Text encoding took 0.0114 ms



.. image:: mobileclip-video-search-with-output_files/mobileclip-video-search-with-output_17_1.png


Convert Model to OpenVINO Intermediate Representation format
------------------------------------------------------------



For best results with OpenVINO, it is recommended to convert the model
to OpenVINO IR format. OpenVINO supports PyTorch via Model conversion
API. To convert the PyTorch model to OpenVINO IR format we will use
``ov.convert_model`` of `model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
The ``ov.convert_model`` Python function returns an OpenVINO Model
object ready to load on the device and start making predictions.

Our model consist from 2 parts - image encoder and text encoder that can
be used separately. Let’s convert each part to OpenVINO.

.. code:: ipython3

    import types
    import torch.nn.functional as F


    def se_block_forward(self, inputs):
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[8, 8])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x

.. code:: ipython3

    import openvino as ov
    import gc

    ov_models_dir = Path("ov_models")
    ov_models_dir.mkdir(exist_ok=True)

    image_encoder_path = ov_models_dir / f"{model_checkpoint.value}_im_encoder.xml"

    if not image_encoder_path.exists():
        if "mobileclip_s" in model_name:
            model.image_encoder.model.conv_exp.se.forward = types.MethodType(se_block_forward, model.image_encoder.model.conv_exp.se)
        model.forward = model.encode_image
        ov_image_encoder = ov.convert_model(
            model,
            example_input=image_tensor,
            input=[-1, 3, image_tensor.shape[2], image_tensor.shape[3]],
        )
        ov.save_model(ov_image_encoder, image_encoder_path)
        del ov_image_encoder
        gc.collect()

    text_encoder_path = ov_models_dir / f"{model_checkpoint.value}_text_encoder.xml"

    if not text_encoder_path.exists():
        model.forward = model.encode_text
        if model_type.value == "Blip2":
            ov_text_encoder = ov.convert_model(model, example_input=text)
        else:
            ov_text_encoder = ov.convert_model(model, example_input=text, input=[-1, text.shape[1]])
        ov.save_model(ov_text_encoder, text_encoder_path)
        del ov_text_encoder
        gc.collect()

    del model
    gc.collect();


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/notebooks/mobileclip-video-search/ml-mobileclip/mobileclip/modules/common/transformer.py:125: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len != self.num_embeddings:


Run OpenVINO model inference
----------------------------



Select device for image encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    core = ov.Core()

    device = device_widget()

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    ov_compiled_image_encoder = core.compile_model(image_encoder_path, device.value)
    ov_compiled_image_encoder(image_tensor);

Select device for text encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    ov_compiled_text_encoder = core.compile_model(text_encoder_path, device.value)
    ov_compiled_text_encoder(text);

Perform search
~~~~~~~~~~~~~~



.. code:: ipython3

    image_encoding_start = time.perf_counter()
    image_features = torch.from_numpy(ov_compiled_image_encoder(image_tensor)[0])
    image_encoding_end = time.perf_counter()
    print(f"Image encoding took {image_encoding_end - image_encoding_start:.3} ms")
    text_encoding_start = time.perf_counter()
    text_features = torch.from_numpy(ov_compiled_text_encoder(text)[0])
    text_encoding_end = time.perf_counter()
    print(f"Text encoding took {text_encoding_end - text_encoding_start:.3} ms")
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    image_probs = image_probs_function(image_features, text_features)
    selected_image = [torch.argmax(image_probs).item()]

    visualize_result(images, input_labels[0], selected_image);


.. parsed-literal::

    Image encoding took 0.0282 ms
    Text encoding took 0.0049 ms



.. image:: mobileclip-video-search-with-output_files/mobileclip-video-search-with-output_28_1.png


(optional) Translation model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since all text embedding models in this notebook natively supports input
in English only, we can insert a translation model in this pipeline to
support searching in Chinese.

-  **opus-mt-zh-en t** - This is a translation model developed by
   Language Technology Research Group at the University of Helsinki. It
   supports Chinese as source Language and English as target Language
   `model card <https://huggingface.co/Helsinki-NLP/opus-mt-zh-en>`__.

.. code:: ipython3

    from pathlib import Path

    cn2en_trans_model_path = "ov_models/cn2en_trans_model"
    cn2en_trans_model_id = "Helsinki-NLP/opus-mt-zh-en"

    if not Path(cn2en_trans_model_path).exists():
        !optimum-cli export openvino --model {cn2en_trans_model_id} --task text2text-generation-with-past --trust-remote-code {cn2en_trans_model_path}


.. parsed-literal::

    2024-11-22 01:36:23.757087: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-22 01:36:23.781523: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
      warnings.warn("Recommended: pip install sacremoses.")
    Moving the following attributes in the config to the generation config: {'max_length': 512, 'num_beams': 6, 'bad_words_ids': [[65000]]}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/marian/modeling_marian.py:207: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/marian/modeling_marian.py:214: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.size() != (bsz, 1, tgt_len, src_len):
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/marian/modeling_marian.py:246: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:88: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if input_shape[-1] > 1 or self.sliding_window is not None:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:164: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/marian/modeling_marian.py:166: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if (
    Exporting tokenizers to OpenVINO is not supported for tokenizers version > 0.19 and openvino version <= 2024.4. Please downgrade to tokenizers version <= 0.19 to export tokenizers to OpenVINO.


.. code:: ipython3

    from transformers import AutoTokenizer
    from optimum.intel import OVModelForSeq2SeqLM

    tr_tokenizer = AutoTokenizer.from_pretrained(cn2en_trans_model_path)
    tr_model = OVModelForSeq2SeqLM.from_pretrained(cn2en_trans_model_path)


.. parsed-literal::

    2024-11-22 01:36:43.187797: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-22 01:36:43.213112: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
      warnings.warn("Recommended: pip install sacremoses.")


Interactive Demo
----------------



In this part, you can try different supported by tutorial models in
searching frames in the video by text query or image. Upload video and
provide text query or reference image for search and model will find the
most relevant frames according to provided query. You can also try
querying in Chinese, and translation model will be triggered
automatically for Chinese-to-English translation. Please note, different
models can require different optimal threshold for search.

.. code:: ipython3

    import altair as alt
    import cv2
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms.functional import to_pil_image, to_tensor
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Resize,
        ToTensor,
    )
    from open_clip.transform import image_transform
    from typing import Optional


    current_device = device.value
    current_model = image_encoder_path.name.split("_im_encoder")[0]

    available_converted_models = [model_file.name.split("_im_encoder")[0] for model_file in ov_models_dir.glob("*_im_encoder.xml")]
    available_devices = list(core.available_devices) + ["AUTO"]

    download_file(
        "https://storage.openvinotoolkit.org/data/test_data/videos/car-detection.mp4",
        directory=sample_path,
    )
    download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4",
        directory=sample_path,
        filename="coco.mp4",
    )


    def is_english(text):
        for char in text:
            if not char.isascii():
                return False
        return True


    def translate(text):
        if tr_tokenizer:
            t = tr_tokenizer(text, return_tensors="pt")
            r = tr_model.generate(**t)
            text = tr_tokenizer.decode(r[0][1:-1])
        return text


    def get_preprocess_probs_tokenizer(model_name):
        if "mobileclip" in model_name:
            resolution = supported_models["MobileCLIP"][model_name]["image_size"]
            resize_size = resolution
            centercrop_size = resolution
            aug_list = [
                Resize(
                    resize_size,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                CenterCrop(centercrop_size),
                ToTensor(),
            ]
            preprocess = Compose(aug_list)
            tokenizer = mobileclip.get_tokenizer(supported_models["MobileCLIP"][model_name]["model_name"])
            image_probs = default_image_probs
        elif "blip2" in model_name:
            from lavis.models import load_model_and_preprocess

            model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=pretrained, is_eval=True)
            model = Blip2Model(model.ln_vision, model.visual_encoder, model.query_tokens, model.Qformer, model.vision_proj, model.text_proj, model.tokenizer)
            preprocess = vis_processors["eval"]
            tokenizer = model.tokenizer
            image_probs = blip2_image_probs
        else:
            model_configs = supported_models["SigLIP"] if "siglip" in model_name else supported_models["CLIP"]
            resize_size = model_configs[model_name]["image_size"]
            preprocess = image_transform((resize_size, resize_size), is_train=False, resize_mode="longest")
            tokenizer = open_clip.get_tokenizer(model_configs[model_name]["model_name"])
            image_probs = default_image_probs

        return preprocess, image_probs, tokenizer


    def run(
        path: str,
        text_search: str,
        image_search: Optional[Image.Image],
        model_name: str,
        device: str,
        thresh: float,
        stride: int,
        batch_size: int,
    ):
        assert path, "An input video should be provided"
        assert text_search is not None or image_search is not None, "A text or image query should be provided"
        global current_model
        global current_device
        global preprocess
        global tokenizer
        global ov_compiled_image_encoder
        global ov_compiled_text_encoder
        global image_probs_function

        if current_model != model_name or device != current_device:
            ov_compiled_image_encoder = core.compile_model(ov_models_dir / f"{model_name}_im_encoder.xml", device)
            ov_compiled_text_encoder = core.compile_model(ov_models_dir / f"{model_name}_text_encoder.xml", device)
            preprocess, image_probs_function, tokenizer = get_preprocess_probs_tokenizer(model_name)
            current_model = model_name
            current_device = device
        # Load video
        dataset = LoadVideo(path, transforms=preprocess, vid_stride=stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Get image query features
        if image_search:
            image = preprocess(image_search).unsqueeze(0)
            query_features = torch.from_numpy(ov_compiled_image_encoder(image)[0])
            query_features /= query_features.norm(dim=-1, keepdim=True)
        # Get text query features
        else:
            if not is_english(text_search):
                text_search = translate(text_search)
                print(f"Translated input text: {text_search}")
            # Tokenize search phrase
            text = tokenizer([text_search])
            # Encode text query
            query_features = torch.from_numpy(ov_compiled_text_encoder(text)[0])
            query_features /= query_features.norm(dim=-1, keepdim=True)
        # Encode each frame and compare with query features
        matches = []
        matches_probs = []
        res = pd.DataFrame(columns=["Frame", "Timestamp", "Similarity"])
        for image, orig, frame, timestamp in dataloader:
            with torch.no_grad():
                image_features = torch.from_numpy(ov_compiled_image_encoder(image)[0])

            image_features /= image_features.norm(dim=-1, keepdim=True)
            probs = image_probs_function(image_features, query_features)
            probs = probs.cpu().numpy().squeeze(1) if "blip2" in model_name else probs[0]
            # Save frame similarity values
            df = pd.DataFrame(
                {
                    "Frame": frame.tolist(),
                    "Timestamp": torch.round(timestamp / 1000, decimals=2).tolist(),
                    "Similarity": probs.tolist(),
                }
            )
            res = pd.concat([res, df])

            # Check if frame is over threshold
            for i, p in enumerate(probs):
                if p > thresh:
                    matches.append(to_pil_image(orig[i]))
                    matches_probs.append(p)

            print(f"Frames: {frame.tolist()} - Probs: {probs}")

        # Create plot of similarity values
        lines = (
            alt.Chart(res)
            .mark_line(color="firebrick")
            .encode(
                alt.X("Timestamp", title="Timestamp (seconds)"),
                alt.Y("Similarity", scale=alt.Scale(zero=False)),
            )
        ).properties(width=600)
        rule = alt.Chart().mark_rule(strokeDash=[6, 3], size=2).encode(y=alt.datum(thresh))

        selected_frames = np.argsort(-1 * np.array(matches_probs))[:20]
        matched_sorted_frames = [matches[idx] for idx in selected_frames]

        return (
            lines + rule,
            matched_sorted_frames,
        )  # Only return up to 20 images to not crash the UI


    class LoadVideo(Dataset):
        def __init__(self, path, transforms, vid_stride=1):
            self.transforms = transforms
            self.vid_stride = vid_stride
            self.cur_frame = 0
            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

        def __getitem__(self, _):
            # Read video
            # Skip over frames
            for _ in range(self.vid_stride):
                self.cap.grab()
                self.cur_frame += 1

            # Read frame
            _, img = self.cap.retrieve()
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            # Convert to PIL
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(np.uint8(img))

            # Apply transforms
            img_t = self.transforms(img)

            return img_t, to_tensor(img), self.cur_frame, timestamp

        def __len__(self):
            return self.total_frames



.. parsed-literal::

    data/car-detection.mp4:   0%|          | 0.00/2.68M [00:00<?, ?B/s]



.. parsed-literal::

    data/coco.mp4:   0%|          | 0.00/877k [00:00<?, ?B/s]


.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/mobileclip-video-search/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    from gradio_helper import make_demo, Option

    demo = make_demo(
        run=run,
        model_option=Option(choices=available_converted_models, value=model_checkpoint.value),
        device_option=Option(choices=available_devices, value=device.value),
    )

    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







