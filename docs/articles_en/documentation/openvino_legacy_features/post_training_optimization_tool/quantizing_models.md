# Quantizing Models {#pot_default_quantization_usage}


@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   DefaultQuantization Method <pot_compression_algorithms_quantization_default_README>


This guide describes how to apply model quantization with the Default Quantization method without accuracy control, using an unannotated dataset. 
To use this method, create a Python script using an API of Post-Training Optimization Tool (POT) and implement data preparation logic and quantization pipeline. 
If you are not familiar with Python, try :doc:`command-line interface <pot_compression_cli_README>` of POT which is designed to quantize models from 
OpenVINO `Model Zoo <https://github.com/openvinotoolkit/open_model_zoo>`__. The figure below shows the common workflow of the quantization script implemented with POT API.

.. image:: _static/images/default_quantization_flow.svg

The script should include three basic steps:

1. Prepare data and dataset interface.
2. Select quantization parameters.
3. Define and run the quantization process.

Prepare data and dataset interface
##################################

In most cases, it is required to implement only the ``openvino.tools.pot.DataLoader`` interface, which allows acquiring data from a dataset and applying model-specific pre-processing providing access by index. Any implementation should override the following methods: 

* The ``__len__()``, returns the size of the dataset.
* The ``__getitem__()``, provides access to the data by index in the range of 0 to ``len(self)``. It can also encapsulate the logic of model-specific pre-processing. This method should return data in the ``(data, annotation)`` format, in which:

  * The ``data`` is the input that is passed to the model at inference so that it should be properly preprocessed. It can be either the ``numpy.array`` object or a dictionary, where the key is the name of the model input and the value is ``numpy.array`` which corresponds to this input.
  * The ``annotation`` is not used by the Default Quantization method. Therefore, this object can be ``None`` in this case.

Framework data loading classes can be wrapped by the ``openvino.tools.pot.DataLoader`` interface which is usually straightforward. 
For example, the ``torch.utils.data.Dataset`` has a similar interface as the ``openvino.tools.pot.DataLoader``, 
so that its TorchVision implementations can be easily wrapped by POT API.

.. note::

   Model-specific preprocessing (for example, mean/scale normalization), can be embedded into the model at the conversion step, using Model Optimizer. This should be considered during the implementation of the DataLoader interface to avoid "double" normalization, which can lead to the loss of accuracy after optimization.


The example code below defines the ``DataLoader`` for three popular use cases: images, text, and audio.


.. tab-set::

    .. tab-item:: Images
        :sync: images

        .. doxygensnippet:: tools/pot/docs/code/data_loaders.py
            :language: python
            :fragment: image_loader

    .. tab-item:: Text
        :sync: text

        .. doxygensnippet:: tools/pot/docs/code/data_loaders.py
           :language: python
           :fragment: text_loader

    .. tab-item:: Audio
        :sync: audio

        .. doxygensnippet:: tools/pot/docs/code/data_loaders.py
           :language: python
           :fragment: audio_loader


Select quantization parameters
##############################

Default Quantization algorithm has mandatory and optional parameters which are defined as a dictionary:

.. code-block:: py
   :force:

   {
       "name": "DefaultQuantization",
       "params": {
           "target_device": "ANY",
           "stat_subset_size": 300,
           "stat_batch_size": 1
       },
   }


* ``"target_device"`` - the following options are available:

  * ``"ANY"`` (or ``"CPU"``) -  default option to quantize models for CPU, GPU, or NPU
  * ``"CPU_SPR"`` -  to quantize models for CPU SPR (4th Generation Intel® Xeon® Scalable processor family)
  * ``"GNA"``, ``"GNA3"``, ``"GNA3.5"`` - to quantize models for GNA devices respectively.

* ``"stat_subset_size"`` - size of the data subset to calculate activations statistics used for quantization. The whole dataset is used if no parameter is specified. It is recommended to use not less than 300 samples.
* ``"stat_batch_size"`` - size of the batch to calculate activations statistics used for quantization. 1 if no parameter is specified.

For full specification, see the :doc:`Default Quantization method <pot_compression_algorithms_quantization_default_README>`.

Run quantization
####################

POT API provides methods to load and save model objects from OpenVINO Intermediate Representation: the ``load_model`` and ``save_model``. It also has a concept of the ``Pipeline`` that sequentially applies specified optimization methods to the model. The ``create_pipeline`` method is used to instantiate a ``Pipeline`` object.
An example code below shows a basic quantization workflow:


.. code-block:: py
   :force:

   from openvino.tools.pot import IEEngine
   from openvino.tools.pot import load_model, save_model
   from openvino.tools.pot import compress_model_weights
   from openvino.tools.pot import create_pipeline

   # Model config specifies the name of the model and paths to .xml and .bin files of the model.
   model_config = 
   {
       "model_name": "model",
       "model": path_to_xml,
       "weights": path_to_bin,
   }

   # Engine config.
   engine_config = {"device": "CPU"}

   algorithms = [
       {
           "name": "DefaultQuantization",
           "params": {
               "target_device": "ANY",
               "stat_subset_size": 300,
               "stat_batch_size": 1
           },
       }
   ]

   # Step 1: Implement and create a user data loader.
   data_loader = ImageLoader("<path_to_images>")

   # Step 2: Load a model.
   model = load_model(model_config=model_config)

   # Step 3: Initialize the engine for metric calculation and statistics collection.
   engine = IEEngine(config=engine_config, data_loader=data_loader)

   # Step 4: Create a pipeline of compression algorithms and run it.
   pipeline = create_pipeline(algorithms, engine)
   compressed_model = pipeline.run(model=model)

   # Step 5 (Optional): Compress model weights to quantized precision
   #                     to reduce the size of the final .bin file.
   compress_model_weights(compressed_model)

   # Step 6: Save the compressed model to the desired path.
   # Set save_path to the directory where the model should be saved.
   compressed_model_paths = save_model(
       model=compressed_model,
       save_path="optimized_model",
       model_name="optimized_model",
   )


The output of the script is the quantized model that can be used for inference in the same way as the original full-precision model.

If high degradation of accuracy occurs after applying the Default Quantization method, 
it is recommended to follow the tips from :doc:`Quantization Best Practices <pot_docs_BestPractices>` 
article or use :doc:`Accuracy-aware Quantization <pot_accuracyaware_usage>` method.

Quantizing cascaded models
##########################

When the optimized model is a cascaded one (consists of several submodels, for example, MT-CNN), it will be necessary to implement a complex inference pipeline that can properly handle different submodels and data flow between them. POT API provides the ``Engine`` interface for this purpose, which allows customization of the inference logic. However, it is recommended to inherit from ``IEEngine`` helper class that already contains all the logic required to do the inference based on OpenVINO Python API. For more details, see the following :doc:`example <pot_example_face_detection_README>`.

Examples
####################

* Tutorials:

  * `Quantization of Image Classification model <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino>`__
  * `Quantization of Object Detection model from Model Zoo <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-yolov5-quantization-migration>`__
  * `Quantization of Segmentation model for medical data <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/110-ct-segmentation-quantize>`__
  * `Quantization of BERT for Text Classification <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert>`__

* Samples:

  * :doc:`Quantization of 3D segmentation model <pot_example_3d_segmentation_README>`
  * :doc:`Quantization of Face Detection model <pot_example_face_detection_README>`
  * :doc:`Quantization of speech model for GNA device <pot_example_speech_README>`


@endsphinxdirective
