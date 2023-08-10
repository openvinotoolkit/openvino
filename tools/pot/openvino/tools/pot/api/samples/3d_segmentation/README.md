# Quantizing 3D Segmentation Model {#pot_example_3d_segmentation_README}

@sphinxdirective

This example demonstrates the use of the :doc:`Post-training Optimization Tool API <pot_compression_api_README>` for the task of quantizing a 3D segmentation model.
The `Brain Tumor Segmentation <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/brain-tumor-segmentation-0002>`__ model from PyTorch is used for this purpose. A custom ``DataLoader`` is created to load images in NIfTI format from the `Medical Segmentation Decathlon BRATS 2017 <http://medicaldecathlon.com/>`__ dataset for 3D semantic segmentation task and the implementation of the Dice Index metric is used for the model evaluation. In addition, this example demonstrates how one can use image metadata obtained during image reading and preprocessing to post-process the model raw output. The code of the example is available on `GitHub <https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/3d_segmentation>`__.

How to Prepare the Data
#######################

To run this example, you will need to download the Brain Tumors 2017 part of the Medical Segmentation Decathlon image database http://medicaldecathlon.com/.
3D MRI data in NIfTI format can be found in the ``imagesTr`` folder, and segmentation masks are in ``labelsTr``.

How to Run the Example
######################

1. Launch :doc:`Model Downloader <omz_tools_downloader>` tool to download ``brain-tumor-segmentation-0002`` model from the Open Model Zoo repository.

   .. code-block:: sh

      omz_downloader --name brain-tumor-segmentation-0002


2. Launch :doc:`Model Converter <omz_tools_downloader>` tool to generate Intermediate Representation (IR) files for the model:

   .. code-block:: sh

      omz_converter --name brain-tumor-segmentation-0002


3. Launch the example script from the example directory:

   .. code-block:: sh

      python3 ./3d_segmentation_example.py -m <PATH_TO_IR_XML> -d <BraTS_2017/imagesTr> --mask-dir <BraTS_2017/labelsTr>


   Optional: you can specify .bin file of IR directly using the ``-w``, ``--weights`` options.

@endsphinxdirective