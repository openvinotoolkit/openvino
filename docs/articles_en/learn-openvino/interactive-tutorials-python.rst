.. {#tutorials}

Interactive Tutorials (Python)
==============================

.. _notebook tutorials:

.. meta::
   :description: Run Python tutorials on Jupyter notebooks to learn how to use OpenVINO™
                 toolkit for optimized deep learning inference.


.. toctree::
   :maxdepth: 2
   :caption: Notebooks
   :hidden:

<<<<<<< HEAD
   notebooks_installation
=======
   interactive-tutorials-python/notebooks-installation
   interactive-tutorials-python/notebooks-section-0-first-steps
   interactive-tutorials-python/notebooks-section-1-convert-and-optimize
   interactive-tutorials-python/notebooks-section-2-model-demos
   interactive-tutorials-python/notebooks-section-3-model-training
   interactive-tutorials-python/notebooks-section-4-live-demos
>>>>>>> 93825cf73c ([DOCS] Paths fix rebase)


Jupyter notebooks show how to use various OpenVINO features to run optimized deep learning
inference with Python. Notebooks with |binder logo| and |colab logo| buttons can be run in the
browser, no installation required. Just choose a tutorial and click the button.

`Binder <https://mybinder.org/>`__ and `Google Colab <https://colab.research.google.com/>`__
are free online services with limited resources. For the best performance
and more control, you should run the notebooks locally. Follow the
:doc:`Installation Guide <notebooks_installation>` in order to get information
on how to run and manage the notebooks on your system.

.. raw:: html

<<<<<<< HEAD
   <script type="module" crossorigin src="https://openvinotoolkit.github.io/openvino_notebooks/assets/embedded.js"></script>
   <iframe id="notebooks-selector" src="https://openvinotoolkit.github.io/openvino_notebooks/" style="width: 100%; border: none;" title="OpenVINO™ Notebooks - Jupyter notebook tutorials for OpenVINO™"></iframe>


.. note::
   If you have any issues with the notebooks, refer to the **Troubleshooting** and **FAQ**
   sections in the :doc:`Installation Guide <notebooks_installation>` or start a GitHub
=======
.. note::

   `Binder <https://mybinder.org/>`__ and `Google Colab <https://colab.research.google.com/>`__
   are free online services with limited ../about-openvino/additional-resources. For the best performance
   and more control, you should run the notebooks locally. Follow the
   :doc:`Installation Guide <interactive-tutorials-python/notebooks-installation>` in order to get information
   on how to run and manage the notebooks on your machine.


More examples along with additional details regarding OpenVINO Notebooks are available in
OpenVINO™ Notebooks `Github Repository. <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md>`__

The Jupyter notebooks are categorized into following classes:

-  :doc:`First steps with OpenVINO <interactive-tutorials-python/notebooks-section-0-first-steps>`
-  :doc:`Convert & Optimize <interactive-tutorials-python/notebooks-section-1-convert-and-optimize>`
-  :doc:`Model Demos <interactive-tutorials-python/notebooks-section-2-model-demos>`
-  :doc:`Model Training <interactive-tutorials-python/notebooks-section-3-model-training>`
-  :doc:`Live Demos <interactive-tutorials-python/notebooks-section-4-live-demos>`


Below you will find a selection of recommended tutorials that demonstrate inference on a particular model. These tutorials are guaranteed to provide a great experience with inference in OpenVINO:


.. showcase::
   :title: 284-openvoice
   :img: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/284-openvoice/284-openvoice.png

   Voice tone cloning with OpenVoice and OpenVINO.

.. showcase::
   :title: 283-photo-maker
   :img: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/283-photo-maker/283-photo-maker.gif

   Text-to-image generation using PhotoMaker and OpenVINO.

.. showcase::
   :title: 281-kosmos2-multimodal-large-language-model
   :img: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/281-kosmos2-multimodal-large-language-model/281-kosmos2-multimodal-large-language-model.png

   Kosmos-2: Multimodal Large Language Model and OpenVINO.

.. showcase::
   :title: 280-depth-anything
   :img: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/280-depth-anything/280-depth-anything.gif

   Depth estimation with DepthAnything and OpenVINO.

.. showcase::
   :title: 279-mobilevlm-language-assistant
   :img: ../_static/images/notebook_eye.png

   Mobile language assistant with MobileVLM and OpenVINO.

.. showcase::
   :title: 278-stable-diffusion-ip-adapter
   :img: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/278-stable-diffusion-ip-adapter/278-stable-diffusion-ip-adapter.png

   Image Generation with Stable Diffusion and IP-Adapter.

.. showcase::
   :title: 275-llm-question-answering
   :img: ../_static/images/notebook_eye.png

   LLM Instruction-following pipeline with OpenVINO.

.. showcase::
   :title: 274-efficient-sam
   :img: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/274-efficient-sam/274-efficient-sam.png

   Object segmentations with EfficientSAM and OpenVINO.

.. showcase::
   :title: 273-stable-zephyr-3b-chatbot
   :img: ../_static/images/notebook_eye.png

   LLM-powered chatbot using Stable-Zephyr-3b and OpenVINO.


.. showcase::
   :title: 272-paint-by-example
   :img: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/272-paint-by-example/272-paint-by-example.png

   Paint by Example using Stable Diffusion and OpenVINO.


.. note::
   If there are any issues while running the notebooks, refer to the **Troubleshooting** and **FAQ** sections in the :doc:`Installation Guide <interactive-tutorials-python/notebooks-installation>` or start a GitHub
>>>>>>> 93825cf73c ([DOCS] Paths fix rebase)
   `discussion <https://github.com/openvinotoolkit/openvino_notebooks/discussions>`__.


Additional Resources
######################

* `OpenVINO™ Notebooks - Github Repository <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md>`_
* `Binder documentation <https://mybinder.readthedocs.io/en/latest/>`_
* `Google Colab <https://colab.research.google.com/>`__


.. |binder logo| image:: _static/images/launch_in_binder.svg
   :class: notebook-badge-p
   :alt: Binder button
.. |colab logo| image:: _static/images/open_in_colab.svg
   :class: notebook-badge-p
   :alt: Google Colab button

