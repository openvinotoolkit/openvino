# Optimization with Simplified Mode {#pot_docs_simplified_mode}

@sphinxdirective

Introduction
####################

Simplified mode is designed to make data preparation for the model optimization process easier. The mode is represented by an implementation of the Engine interface from the POT API. It allows reading the data from an arbitrary folder specified by the user. For more details about POT API, refer to the corresponding :doc:`description <pot_compression_api_README>`. Currently, Simplified mode is available only for image data in PNG or JPEG formats, stored in a single folder. It supports Computer Vision models with a single input or two inputs where the second is "image_info" (Faster R-CNN, Mask R-CNN, etc.).

.. note::

   This mode cannot be used with accuracy-aware methods. There is no way to control accuracy after optimization. Nevertheless, this mode can be helpful to estimate performance benefits when using model optimizations.

Usage
####################

To use the Simplified mode, prepare the data and place it in a separate folder. No other files should be present in this folder.

To apply optimization when there is only a model and no data is available. It is possible to generate a synthetic dataset using Dataset Management Framework (Datumaro) available on `GitHub <https://github.com/openvinotoolkit/datumaro>`__. Currently, data generation is available only for Computer Vision models, it can take time in some cases.

Install Datumaro:

.. code-block:: sh

   pip install datumaro


Create a synthetic dataset with elements of the specified type and shape, and save it to the provided directory.

Usage:

.. code-block:: sh

   datum generate [-h] -o OUTPUT_DIR -k COUNT --shape SHAPE [SHAPE ...]
     [-t {image}] [--overwrite] [--model-dir MODEL_PATH]


Example of generating 300 images with height = 224 and width = 256 and saving them in the ``./dataset`` directory.

.. code-block:: sh

   datum generate  -o ./dataset -k 300 --shape 224 256


After that, ``OUTPUT_DIR`` can be provided to the ``--data-source`` CLI option or to the ``data_source`` config parameter.

There are two options to run POT in the Simplified mode:

* Using command-line options only. Here is an example of 8-bit quantization:

  ``pot -q default -m <path_to_xml> -w <path_to_bin> --engine simplified --data-source <path_to_data>``

* To provide more options, use the corresponding `"engine"` section in the POT configuration file as follows:

  .. code-block:: json

     "engine": {
         "type": "simplified",
         "layout": "NCHW",               // Layout of input data. Supported ["NCHW",
                                         // "NHWC", "CHW", "CWH"] layout
         "data_source": "PATH_TO_SOURCE" // You can specify a path to the directory with images
                                         // Also you can specify template for file names to filter images to load.
                                         // Templates are unix style (this option is valid only in Simplified mode)
     }


A template of the configuration file for 8-bit quantization using Simplified mode can be found `at the following link <https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/simplified_mode_template.json>`__.

For more details about POT usage via CLI, refer to this :doc:`CLI document <pot_compression_cli_README>`.

Additional Resources
####################

* :doc:`Configuration File Description <pot_configs_README>`

@endsphinxdirective