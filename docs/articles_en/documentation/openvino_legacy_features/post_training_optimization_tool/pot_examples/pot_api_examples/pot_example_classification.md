# Quantizing Image Classification Model {#pot_example_classification_README}

@sphinxdirective

This example demonstrates the use of the :doc:`Post-training Optimization Tool API <pot_compression_api_README>` for the task of quantizing a classification model.
The `MobilenetV2 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-1.0-224>`__ model from TensorFlow is used for this purpose.
A custom ``DataLoader`` is created to load the `ImageNet <http://www.image-net.org/>`__ classification dataset and the implementation of Accuracy at top-1 metric is used for the model evaluation. The code of the example is available on `GitHub <https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/classification>`__.

How to Prepare the Data
#######################

To run this example, you need to `download <https://image-net.org/download.php>`__ the validation part of the ImageNet image database and place it in a separate folder, 
which will be later referred to as ``<IMAGES_DIR>``. Annotations to images should be stored in a separate .txt file (``<IMAGENET_ANNOTATION_FILE>``) in the format ``image_name label``.


How to Run the Example
######################

1. Launch :doc:`Model Downloader <omz_tools_downloader>` tool to download ``mobilenet-v2-1.0-224`` model from the Open Model Zoo repository.

   .. code-block:: sh

      omz_downloader --name mobilenet-v2-1.0-224

2. Launch :doc:`Model Converter <omz_tools_downloader>` tool to generate Intermediate Representation (IR) files for the model:

   .. code-block:: sh

      omz_converter --name mobilenet-v2-1.0-224 --mo <PATH_TO_MODEL_OPTIMIZER>/mo.py

3. Launch the example script from the example directory:

   .. code-block:: sh

      python3 ./classification_sample.py -m <PATH_TO_IR_XML> -a <IMAGENET_ANNOTATION_FILE> -d <IMAGES_DIR>

   Optional: you can specify .bin file of IR directly using the ``-w``, ``--weights`` options.

@endsphinxdirective
