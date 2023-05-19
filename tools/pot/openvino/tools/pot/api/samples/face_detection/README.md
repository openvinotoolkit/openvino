# Quantizing Cascaded Face detection Model {#pot_example_face_detection_README}

@sphinxdirective

This example demonstrates the use of the :doc:`Post-training Optimization Tool API <pot_compression_api_README>` for the task of quantizing a face detection model.
The `MTCNN <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mtcnn>`__ model from Caffe is used for this purpose.
A custom ``DataLoader`` is created to load the `WIDER FACE <http://shuoyang1213.me/WIDERFACE/>`__ dataset for a face detection task 
and the implementation of Recall metric is used for the model evaluation. In addition, this example demonstrates how one can implement 
an engine to infer a cascaded (composite) model that is represented by multiple submodels in an OpenVINO™ Intermediate Representation (IR)
and has a complex staged inference pipeline. The code of the example is available on `GitHub <https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/face_detection>`__.

How to Prepare the Data
#######################

To run this example, you need to download the validation part of the Wider Face dataset http://shuoyang1213.me/WIDERFACE/.
Images with faces divided into categories are placed in the ``WIDER_val/images`` folder. 
Annotations in .txt format containing the coordinates of the face bounding boxes of the 
validation part of the dataset can be downloaded separately and are located in the ``wider_face_split/wider_face_val_bbx_gt.txt`` file.

How to Run the Example
######################

1. Launch :doc:`Model Downloader <omz_tools_downloader>` tool to download ``mtcnn`` model from the Open Model Zoo repository.

   .. code-block:: sh

      omz_downloader --name mtcnn*


2. Launch :doc:`Model Converter <omz_tools_downloader>` tool to generate Intermediate Representation (IR) files for the model:

   .. code-block:: sh

      omz_converter --name mtcnn* --mo <PATH_TO_MODEL_OPTIMIZER>/mo.py


3. Launch the example script from the example directory:

   .. code-block:: sh

      python3 ./face_detection_example.py -pm <PATH_TO_IR_XML_OF_PNET_MODEL> 
      -rm <PATH_TO_IR_XML_OF_RNET_MODEL> -om <PATH_TO_IR_XML_OF_ONET_MODEL> -d <WIDER_val/images> -a <wider_face_split/wider_face_val_bbx_gt.txt>


   Optional: you can specify .bin files of corresponding IRs directly using the ``-pw/--pnet-weights``, ``-rw/--rnet-weights`` and ``-ow/--onet-weights`` options.

@endsphinxdirective
