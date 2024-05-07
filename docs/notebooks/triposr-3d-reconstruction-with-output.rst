TripoSR feedforward 3D reconstruction from a single image and OpenVINO
======================================================================

`TripoSR <https://huggingface.co/spaces/stabilityai/TripoSR>`__ is a
state-of-the-art open-source model for fast feedforward 3D
reconstruction from a single image, developed in collaboration between
`Tripo AI <https://www.tripo3d.ai/>`__ and `Stability
AI <https://stability.ai/news/triposr-3d-generation>`__.

You can find `the source code on
GitHub <https://github.com/VAST-AI-Research/TripoSR>`__ and `demo on
HuggingFace <https://huggingface.co/spaces/stabilityai/TripoSR>`__.
Also, you can read the paper `TripoSR: Fast 3D Object Reconstruction
from a Single Image <https://arxiv.org/abs/2403.02151>`__.



Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Get the original model <#get-the-original-model>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Compiling models and prepare
   pipeline <#compiling-models-and-prepare-pipeline>`__
-  `Interactive inference <#interactive-inference>`__

Prerequisites
-------------



.. code:: ipython3

    %pip install -q wheel setuptools pip --upgrade
    %pip install -q "gradio>=4.19" torch rembg trimesh einops omegaconf "transformers>=4.35.0" "openvino>=2024.0.0" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/tatsy/torchmcubes.git"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    WARNING: typer 0.12.3 does not provide the extra 'all'


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import sys
    from pathlib import Path

    if not Path("TripoSR").exists():
        !git clone https://huggingface.co/spaces/stabilityai/TripoSR

    sys.path.append("TripoSR")


.. parsed-literal::

    Cloning into 'TripoSR'...


.. parsed-literal::

    remote: Enumerating objects: 117, done.[K
    remote: Counting objects:   0% (1/117)[K
remote: Counting objects:   1% (2/117)[K
remote: Counting objects:   2% (3/117)[K
remote: Counting objects:   3% (4/117)[K
remote: Counting objects:   4% (5/117)[K
remote: Counting objects:   5% (6/117)[K
remote: Counting objects:   6% (8/117)[K
remote: Counting objects:   7% (9/117)[K
remote: Counting objects:   8% (10/117)[K
remote: Counting objects:   9% (11/117)[K
remote: Counting objects:  10% (12/117)[K
remote: Counting objects:  11% (13/117)[K
remote: Counting objects:  12% (15/117)[K
remote: Counting objects:  13% (16/117)[K
remote: Counting objects:  14% (17/117)[K
remote: Counting objects:  15% (18/117)[K
remote: Counting objects:  16% (19/117)[K
remote: Counting objects:  17% (20/117)[K
remote: Counting objects:  18% (22/117)[K
remote: Counting objects:  19% (23/117)[K
remote: Counting objects:  20% (24/117)[K
remote: Counting objects:  21% (25/117)[K
remote: Counting objects:  22% (26/117)[K
remote: Counting objects:  23% (27/117)[K
remote: Counting objects:  24% (29/117)[K
remote: Counting objects:  25% (30/117)[K
remote: Counting objects:  26% (31/117)[K
remote: Counting objects:  27% (32/117)[K
remote: Counting objects:  28% (33/117)[K
remote: Counting objects:  29% (34/117)[K
remote: Counting objects:  30% (36/117)[K
remote: Counting objects:  31% (37/117)[K
remote: Counting objects:  32% (38/117)[K
remote: Counting objects:  33% (39/117)[K
remote: Counting objects:  34% (40/117)[K
remote: Counting objects:  35% (41/117)[K
remote: Counting objects:  36% (43/117)[K
remote: Counting objects:  37% (44/117)[K
remote: Counting objects:  38% (45/117)[K
remote: Counting objects:  39% (46/117)[K
remote: Counting objects:  40% (47/117)[K
remote: Counting objects:  41% (48/117)[K
remote: Counting objects:  42% (50/117)[K
remote: Counting objects:  43% (51/117)[K
remote: Counting objects:  44% (52/117)[K
remote: Counting objects:  45% (53/117)[K
remote: Counting objects:  46% (54/117)[K
remote: Counting objects:  47% (55/117)[K
remote: Counting objects:  48% (57/117)[K
remote: Counting objects:  49% (58/117)[K
remote: Counting objects:  50% (59/117)[K
remote: Counting objects:  51% (60/117)[K
remote: Counting objects:  52% (61/117)[K
remote: Counting objects:  53% (63/117)[K
remote: Counting objects:  54% (64/117)[K
remote: Counting objects:  55% (65/117)[K
remote: Counting objects:  56% (66/117)[K
remote: Counting objects:  57% (67/117)[K
remote: Counting objects:  58% (68/117)[K
remote: Counting objects:  59% (70/117)[K
remote: Counting objects:  60% (71/117)[K
remote: Counting objects:  61% (72/117)[K
remote: Counting objects:  62% (73/117)[K
remote: Counting objects:  63% (74/117)[K
remote: Counting objects:  64% (75/117)[K
remote: Counting objects:  65% (77/117)[K
remote: Counting objects:  66% (78/117)[K
remote: Counting objects:  67% (79/117)[K
remote: Counting objects:  68% (80/117)[K
remote: Counting objects:  69% (81/117)[K
remote: Counting objects:  70% (82/117)[K
remote: Counting objects:  71% (84/117)[K
remote: Counting objects:  72% (85/117)[K
remote: Counting objects:  73% (86/117)[K
remote: Counting objects:  74% (87/117)[K
remote: Counting objects:  75% (88/117)[K
remote: Counting objects:  76% (89/117)[K
remote: Counting objects:  77% (91/117)[K
remote: Counting objects:  78% (92/117)[K
remote: Counting objects:  79% (93/117)[K
remote: Counting objects:  80% (94/117)[K
remote: Counting objects:  81% (95/117)[K
remote: Counting objects:  82% (96/117)[K
remote: Counting objects:  83% (98/117)[K
remote: Counting objects:  84% (99/117)[K
remote: Counting objects:  85% (100/117)[K
remote: Counting objects:  86% (101/117)[K
remote: Counting objects:  87% (102/117)[K
remote: Counting objects:  88% (103/117)[K
remote: Counting objects:  89% (105/117)[K
remote: Counting objects:  90% (106/117)[K
remote: Counting objects:  91% (107/117)[K
remote: Counting objects:  92% (108/117)[K
remote: Counting objects:  93% (109/117)[K
remote: Counting objects:  94% (110/117)[K
remote: Counting objects:  95% (112/117)[K
remote: Counting objects:  96% (113/117)[K
remote: Counting objects:  97% (114/117)[K
remote: Counting objects:  98% (115/117)[K
remote: Counting objects:  99% (116/117)[K
remote: Counting objects: 100% (117/117)[K
remote: Counting objects: 100% (117/117), done.[K
    remote: Compressing objects:   1% (1/77)[K
remote: Compressing objects:   2% (2/77)[K
remote: Compressing objects:   3% (3/77)[K
remote: Compressing objects:   5% (4/77)[K
remote: Compressing objects:   6% (5/77)[K
remote: Compressing objects:   7% (6/77)[K
remote: Compressing objects:   9% (7/77)[K
remote: Compressing objects:  10% (8/77)[K
remote: Compressing objects:  11% (9/77)[K
remote: Compressing objects:  12% (10/77)[K
remote: Compressing objects:  14% (11/77)[K
remote: Compressing objects:  15% (12/77)[K
remote: Compressing objects:  16% (13/77)[K
remote: Compressing objects:  18% (14/77)[K
remote: Compressing objects:  19% (15/77)[K
remote: Compressing objects:  20% (16/77)[K
remote: Compressing objects:  22% (17/77)[K
remote: Compressing objects:  23% (18/77)[K
remote: Compressing objects:  24% (19/77)[K
remote: Compressing objects:  25% (20/77)[K
remote: Compressing objects:  27% (21/77)[K
remote: Compressing objects:  28% (22/77)[K
remote: Compressing objects:  29% (23/77)[K
remote: Compressing objects:  31% (24/77)[K
remote: Compressing objects:  32% (25/77)[K
remote: Compressing objects:  33% (26/77)[K
remote: Compressing objects:  35% (27/77)[K
remote: Compressing objects:  36% (28/77)[K
remote: Compressing objects:  37% (29/77)[K
remote: Compressing objects:  38% (30/77)[K
remote: Compressing objects:  40% (31/77)[K
remote: Compressing objects:  41% (32/77)[K
remote: Compressing objects:  42% (33/77)[K
remote: Compressing objects:  44% (34/77)[K
remote: Compressing objects:  45% (35/77)[K
remote: Compressing objects:  46% (36/77)[K
remote: Compressing objects:  48% (37/77)[K
remote: Compressing objects:  49% (38/77)[K
remote: Compressing objects:  50% (39/77)[K
remote: Compressing objects:  51% (40/77)[K
remote: Compressing objects:  53% (41/77)[K
remote: Compressing objects:  54% (42/77)[K
remote: Compressing objects:  55% (43/77)[K
remote: Compressing objects:  57% (44/77)[K
remote: Compressing objects:  58% (45/77)[K
remote: Compressing objects:  59% (46/77)[K
remote: Compressing objects:  61% (47/77)[K
remote: Compressing objects:  62% (48/77)[K
remote: Compressing objects:  63% (49/77)[K
remote: Compressing objects:  64% (50/77)[K
remote: Compressing objects:  66% (51/77)[K
remote: Compressing objects:  67% (52/77)[K
remote: Compressing objects:  68% (53/77)[K
remote: Compressing objects:  70% (54/77)[K
remote: Compressing objects:  71% (55/77)[K
remote: Compressing objects:  72% (56/77)[K
remote: Compressing objects:  74% (57/77)[K
remote: Compressing objects:  75% (58/77)[K
remote: Compressing objects:  76% (59/77)[K
remote: Compressing objects:  77% (60/77)[K
remote: Compressing objects:  79% (61/77)[K
remote: Compressing objects:  80% (62/77)[K
remote: Compressing objects:  81% (63/77)[K
remote: Compressing objects:  83% (64/77)[K
remote: Compressing objects:  84% (65/77)[K
remote: Compressing objects:  85% (66/77)[K
remote: Compressing objects:  87% (67/77)[K
remote: Compressing objects:  88% (68/77)[K
remote: Compressing objects:  89% (69/77)[K
remote: Compressing objects:  90% (70/77)[K
remote: Compressing objects:  92% (71/77)[K
remote: Compressing objects:  93% (72/77)[K
remote: Compressing objects:  94% (73/77)[K
remote: Compressing objects:  96% (74/77)[K
remote: Compressing objects:  97% (75/77)[K
remote: Compressing objects:  98% (76/77)[K
remote: Compressing objects: 100% (77/77)[K
remote: Compressing objects: 100% (77/77), done.[K


.. parsed-literal::

    Receiving objects:   0% (1/117)
Receiving objects:   1% (2/117)
Receiving objects:   2% (3/117)
Receiving objects:   3% (4/117)
Receiving objects:   4% (5/117)
Receiving objects:   5% (6/117)
Receiving objects:   6% (8/117)
Receiving objects:   7% (9/117)
Receiving objects:   8% (10/117)
Receiving objects:   9% (11/117)
Receiving objects:  10% (12/117)
Receiving objects:  11% (13/117)
Receiving objects:  12% (15/117)
Receiving objects:  13% (16/117)
Receiving objects:  14% (17/117)
Receiving objects:  15% (18/117)
Receiving objects:  16% (19/117)
Receiving objects:  17% (20/117)
Receiving objects:  18% (22/117)
Receiving objects:  19% (23/117)
Receiving objects:  20% (24/117)
Receiving objects:  21% (25/117)
Receiving objects:  22% (26/117)
Receiving objects:  23% (27/117)
Receiving objects:  24% (29/117)
Receiving objects:  25% (30/117)
Receiving objects:  26% (31/117)
Receiving objects:  27% (32/117)
Receiving objects:  28% (33/117)
Receiving objects:  29% (34/117)
Receiving objects:  30% (36/117)
Receiving objects:  31% (37/117)
Receiving objects:  32% (38/117)
Receiving objects:  33% (39/117)
Receiving objects:  34% (40/117)
Receiving objects:  35% (41/117)
Receiving objects:  36% (43/117)
Receiving objects:  37% (44/117)
Receiving objects:  38% (45/117)
Receiving objects:  39% (46/117)
Receiving objects:  40% (47/117)
Receiving objects:  41% (48/117)
Receiving objects:  42% (50/117)
Receiving objects:  43% (51/117)
Receiving objects:  44% (52/117)
Receiving objects:  45% (53/117)
Receiving objects:  46% (54/117)

.. parsed-literal::

    Receiving objects:  47% (55/117)
Receiving objects:  48% (57/117)
Receiving objects:  49% (58/117)

.. parsed-literal::

    Receiving objects:  50% (59/117)

.. parsed-literal::

    Receiving objects:  51% (60/117)
Receiving objects:  52% (61/117)

.. parsed-literal::

    Receiving objects:  53% (63/117)
Receiving objects:  54% (64/117)
Receiving objects:  55% (65/117)
Receiving objects:  56% (66/117)
remote: Total 117 (delta 38), reused 117 (delta 38), pack-reused 0[K
    Receiving objects:  57% (67/117)
Receiving objects:  58% (68/117)
Receiving objects:  59% (70/117)
Receiving objects:  60% (71/117)
Receiving objects:  61% (72/117)
Receiving objects:  62% (73/117)
Receiving objects:  63% (74/117)
Receiving objects:  64% (75/117)
Receiving objects:  65% (77/117)
Receiving objects:  66% (78/117)
Receiving objects:  67% (79/117)
Receiving objects:  68% (80/117)
Receiving objects:  69% (81/117)
Receiving objects:  70% (82/117)
Receiving objects:  71% (84/117)
Receiving objects:  72% (85/117)
Receiving objects:  73% (86/117)
Receiving objects:  74% (87/117)
Receiving objects:  75% (88/117)
Receiving objects:  76% (89/117)
Receiving objects:  77% (91/117)
Receiving objects:  78% (92/117)
Receiving objects:  79% (93/117)
Receiving objects:  80% (94/117)
Receiving objects:  81% (95/117)
Receiving objects:  82% (96/117)
Receiving objects:  83% (98/117)
Receiving objects:  84% (99/117)
Receiving objects:  85% (100/117)
Receiving objects:  86% (101/117)
Receiving objects:  87% (102/117)
Receiving objects:  88% (103/117)
Receiving objects:  89% (105/117)
Receiving objects:  90% (106/117)
Receiving objects:  91% (107/117)
Receiving objects:  92% (108/117)
Receiving objects:  93% (109/117)
Receiving objects:  94% (110/117)
Receiving objects:  95% (112/117)
Receiving objects:  96% (113/117)
Receiving objects:  97% (114/117)
Receiving objects:  98% (115/117)
Receiving objects:  99% (116/117)
Receiving objects: 100% (117/117)
Receiving objects: 100% (117/117), 568.99 KiB | 2.54 MiB/s, done.
    Resolving deltas:   0% (0/38)
Resolving deltas:   5% (2/38)
Resolving deltas:  15% (6/38)
Resolving deltas:  18% (7/38)
Resolving deltas:  23% (9/38)
Resolving deltas:  26% (10/38)
Resolving deltas:  31% (12/38)
Resolving deltas:  42% (16/38)
Resolving deltas:  44% (17/38)
Resolving deltas:  47% (18/38)
Resolving deltas:  50% (19/38)
Resolving deltas:  57% (22/38)
Resolving deltas:  68% (26/38)
Resolving deltas:  73% (28/38)
Resolving deltas:  92% (35/38)
Resolving deltas: 100% (38/38)
Resolving deltas: 100% (38/38), done.


Get the original model
----------------------

.. code:: ipython3

    import os

    from tsr.system import TSR


    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(131072)
    model.to("cpu")




.. parsed-literal::

    TSR(
      (image_tokenizer): DINOSingleImageTokenizer(
        (model): ViTModel(
          (embeddings): ViTEmbeddings(
            (patch_embeddings): ViTPatchEmbeddings(
              (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
            )
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (encoder): ViTEncoder(
            (layer): ModuleList(
              (0-11): 12 x ViTLayer(
                (attention): ViTAttention(
                  (attention): ViTSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                  (output): ViTSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.0, inplace=False)
                  )
                )
                (intermediate): ViTIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ViTOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
                (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              )
            )
          )
          (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (pooler): ViTPooler(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (activation): Tanh()
          )
        )
      )
      (tokenizer): Triplane1DTokenizer()
      (backbone): Transformer1D(
        (norm): GroupNorm(32, 1024, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1024, out_features=1024, bias=True)
        (transformer_blocks): ModuleList(
          (0-15): 16 x BasicTransformerBlock(
            (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=1024, out_features=1024, bias=False)
              (to_k): Linear(in_features=1024, out_features=1024, bias=False)
              (to_v): Linear(in_features=1024, out_features=1024, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1024, out_features=1024, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=1024, out_features=1024, bias=False)
              (to_k): Linear(in_features=768, out_features=1024, bias=False)
              (to_v): Linear(in_features=768, out_features=1024, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1024, out_features=1024, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1024, out_features=8192, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=4096, out_features=1024, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=1024, out_features=1024, bias=True)
      )
      (post_processor): TriplaneUpsampleNetwork(
        (upsample): ConvTranspose2d(1024, 40, kernel_size=(2, 2), stride=(2, 2))
      )
      (decoder): NeRFMLP(
        (layers): Sequential(
          (0): Linear(in_features=120, out_features=64, bias=True)
          (1): SiLU(inplace=True)
          (2): Linear(in_features=64, out_features=64, bias=True)
          (3): SiLU(inplace=True)
          (4): Linear(in_features=64, out_features=64, bias=True)
          (5): SiLU(inplace=True)
          (6): Linear(in_features=64, out_features=64, bias=True)
          (7): SiLU(inplace=True)
          (8): Linear(in_features=64, out_features=64, bias=True)
          (9): SiLU(inplace=True)
          (10): Linear(in_features=64, out_features=64, bias=True)
          (11): SiLU(inplace=True)
          (12): Linear(in_features=64, out_features=64, bias=True)
          (13): SiLU(inplace=True)
          (14): Linear(in_features=64, out_features=64, bias=True)
          (15): SiLU(inplace=True)
          (16): Linear(in_features=64, out_features=64, bias=True)
          (17): SiLU(inplace=True)
          (18): Linear(in_features=64, out_features=4, bias=True)
        )
      )
      (renderer): TriplaneNeRFRenderer()
    )



Convert the model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Define the conversion function for PyTorch modules. We use
``ov.convert_model`` function to obtain OpenVINO Intermediate
Representation object and ``ov.save_model`` function to save it as XML
file.

.. code:: ipython3

    import torch

    import openvino as ov


    def convert(model: torch.nn.Module, xml_path: str, example_input):
        xml_path = Path(xml_path)
        if not xml_path.exists():
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                converted_model = ov.convert_model(model, example_input=example_input)
            ov.save_model(converted_model, xml_path, compress_to_fp16=False)

            # cleanup memory
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()

The original model is a pipeline of several models. There are
``image_tokenizer``, ``tokenizer``, ``backbone`` and ``post_processor``.
``image_tokenizer`` contains ``ViTModel`` that consists of
``ViTPatchEmbeddings``, ``ViTEncoder`` and ``ViTPooler``. ``tokenizer``
is ``Triplane1DTokenizer``, ``backbone`` is ``Transformer1D``,
``post_processor`` is ``TriplaneUpsampleNetwork``. Convert all internal
models one by one.

.. code:: ipython3

    VIT_PATCH_EMBEDDINGS_OV_PATH = Path("models/vit_patch_embeddings_ir.xml")


    class PatchEmbedingWrapper(torch.nn.Module):
        def __init__(self, patch_embeddings):
            super().__init__()
            self.patch_embeddings = patch_embeddings

        def forward(self, pixel_values, interpolate_pos_encoding=True):
            outputs = self.patch_embeddings(pixel_values=pixel_values, interpolate_pos_encoding=True)
            return outputs


    example_input = {
        "pixel_values": torch.rand([1, 3, 512, 512], dtype=torch.float32),
    }

    convert(
        PatchEmbedingWrapper(model.image_tokenizer.model.embeddings.patch_embeddings),
        VIT_PATCH_EMBEDDINGS_OV_PATH,
        example_input,
    )


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/vit/modeling_vit.py:170: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if num_channels != self.num_channels:


.. code:: ipython3

    VIT_ENCODER_OV_PATH = Path("models/vit_encoder_ir.xml")


    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(
            self,
            hidden_states=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        ):
            outputs = self.encoder(
                hidden_states=hidden_states,
            )

            return outputs.last_hidden_state


    example_input = {
        "hidden_states": torch.rand([1, 1025, 768], dtype=torch.float32),
    }

    convert(
        EncoderWrapper(model.image_tokenizer.model.encoder),
        VIT_ENCODER_OV_PATH,
        example_input,
    )

.. code:: ipython3

    VIT_POOLER_OV_PATH = Path("models/vit_pooler_ir.xml")
    convert(
        model.image_tokenizer.model.pooler,
        VIT_POOLER_OV_PATH,
        torch.rand([1, 1025, 768], dtype=torch.float32),
    )

.. code:: ipython3

    TOKENIZER_OV_PATH = Path("models/tokenizer_ir.xml")
    convert(model.tokenizer, TOKENIZER_OV_PATH, torch.tensor(1))

.. code:: ipython3

    example_input = {
        "hidden_states": torch.rand([1, 1024, 3072], dtype=torch.float32),
        "encoder_hidden_states": torch.rand([1, 1025, 768], dtype=torch.float32),
    }

    BACKBONE_OV_PATH = Path("models/backbone_ir.xml")
    convert(model.backbone, BACKBONE_OV_PATH, example_input)

.. code:: ipython3

    POST_PROCESSOR_OV_PATH = Path("models/post_processor_ir.xml")
    convert(
        model.post_processor,
        POST_PROCESSOR_OV_PATH,
        torch.rand([1, 3, 1024, 32, 32], dtype=torch.float32),
    )

Compiling models and prepare pipeline
-------------------------------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import ipywidgets as widgets


    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    compiled_vit_patch_embeddings = core.compile_model(VIT_PATCH_EMBEDDINGS_OV_PATH, device.value)
    compiled_vit_model_encoder = core.compile_model(VIT_ENCODER_OV_PATH, device.value)
    compiled_vit_model_pooler = core.compile_model(VIT_POOLER_OV_PATH, device.value)

    compiled_tokenizer = core.compile_model(TOKENIZER_OV_PATH, device.value)
    compiled_backbone = core.compile_model(BACKBONE_OV_PATH, device.value)
    compiled_post_processor = core.compile_model(POST_PROCESSOR_OV_PATH, device.value)

Let’s create callable wrapper classes for compiled models to allow
interaction with original ``TSR`` class. Note that all of wrapper
classes return ``torch.Tensor``\ s instead of ``np.array``\ s.

.. code:: ipython3

    from collections import namedtuple


    class VitPatchEmdeddingsWrapper(torch.nn.Module):
        def __init__(self, vit_patch_embeddings, model):
            super().__init__()
            self.vit_patch_embeddings = vit_patch_embeddings
            self.projection = model.projection

        def forward(self, pixel_values, interpolate_pos_encoding=False):
            inputs = {
                "pixel_values": pixel_values,
            }
            outs = self.vit_patch_embeddings(inputs)[0]

            return torch.from_numpy(outs)


    class VitModelEncoderWrapper(torch.nn.Module):
        def __init__(self, vit_model_encoder):
            super().__init__()
            self.vit_model_encoder = vit_model_encoder

        def forward(
            self,
            hidden_states,
            head_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        ):
            inputs = {
                "hidden_states": hidden_states.detach().numpy(),
            }

            outs = self.vit_model_encoder(inputs)
            outputs = namedtuple("BaseModelOutput", ("last_hidden_state", "hidden_states", "attentions"))

            return outputs(torch.from_numpy(outs[0]), None, None)


    class VitModelPoolerWrapper(torch.nn.Module):
        def __init__(self, vit_model_pooler):
            super().__init__()
            self.vit_model_pooler = vit_model_pooler

        def forward(self, hidden_states):
            outs = self.vit_model_pooler(hidden_states.detach().numpy())[0]

            return torch.from_numpy(outs)


    class TokenizerWrapper(torch.nn.Module):
        def __init__(self, tokenizer, model):
            super().__init__()
            self.tokenizer = tokenizer
            self.detokenize = model.detokenize

        def forward(self, batch_size):
            outs = self.tokenizer(batch_size)[0]

            return torch.from_numpy(outs)


    class BackboneWrapper(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, hidden_states, encoder_hidden_states):
            inputs = {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
            }

            outs = self.backbone(inputs)[0]

            return torch.from_numpy(outs)


    class PostProcessorWrapper(torch.nn.Module):
        def __init__(self, post_processor):
            super().__init__()
            self.post_processor = post_processor

        def forward(self, triplanes):
            outs = self.post_processor(triplanes)[0]

            return torch.from_numpy(outs)

Replace all models in the original model by wrappers instances:

.. code:: ipython3

    model.image_tokenizer.model.embeddings.patch_embeddings = VitPatchEmdeddingsWrapper(
        compiled_vit_patch_embeddings,
        model.image_tokenizer.model.embeddings.patch_embeddings,
    )
    model.image_tokenizer.model.encoder = VitModelEncoderWrapper(compiled_vit_model_encoder)
    model.image_tokenizer.model.pooler = VitModelPoolerWrapper(compiled_vit_model_pooler)

    model.tokenizer = TokenizerWrapper(compiled_tokenizer, model.tokenizer)
    model.backbone = BackboneWrapper(compiled_backbone)
    model.post_processor = PostProcessorWrapper(compiled_post_processor)

Interactive inference
---------------------



.. code:: ipython3

    import tempfile

    import gradio as gr
    import numpy as np
    import rembg
    from PIL import Image

    from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation


    rembg_session = rembg.new_session()


    def check_input_image(input_image):
        if input_image is None:
            raise gr.Error("No image uploaded!")


    def preprocess(input_image, do_remove_background, foreground_ratio):
        def fill_background(image):
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            return image

        if do_remove_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = fill_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = fill_background(image)
        return image


    def generate(image):
        scene_codes = model(image, "cpu")  # the device is provided for the image processor
        mesh = model.extract_mesh(scene_codes)[0]
        mesh = to_gradio_3d_orientation(mesh)
        mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        mesh.export(mesh_path.name)
        return mesh_path.name


    with gr.Blocks() as demo:
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)
                with gr.Row():
                    with gr.Group():
                        do_remove_background = gr.Checkbox(label="Remove Background", value=True)
                        foreground_ratio = gr.Slider(
                            label="Foreground Ratio",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                        )
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")
            with gr.Column():
                with gr.Tab("Model"):
                    output_model = gr.Model3D(
                        label="Output Model",
                        interactive=False,
                    )
        with gr.Row(variant="panel"):
            gr.Examples(
                examples=[os.path.join("TripoSR/examples", img_name) for img_name in sorted(os.listdir("TripoSR/examples"))],
                inputs=[input_image],
                outputs=[processed_image, output_model],
                label="Examples",
                examples_per_page=20,
            )
        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess,
            inputs=[input_image, do_remove_background, foreground_ratio],
            outputs=[processed_image],
        ).success(
            fn=generate,
            inputs=[processed_image],
            outputs=[output_model],
        )

    try:
        demo.launch(debug=False, height=680)
    except Exception:
        demo.queue().launch(share=True, debug=False, height=680)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







