.. {#openvino_docs_MO_DG_prepare_model_convert_model_mxnet_specific_Convert_GluonCV_Models}

Converting MXNet GluonCV Models
===============================


.. meta::
   :description: Learn how to convert GluonCV models
                 from MXNet to the OpenVINO Intermediate Representation.


.. warning::

   Note that OpenVINO support for Apache MXNet is currently being deprecated and will be removed entirely in the future.

This article provides the instructions and examples on how to convert `GluonCV SSD and YOLO-v3 models <https://gluon-cv.mxnet.io/model_zoo/detection.html>`__ to IR.

1. Choose the topology available from the `GluonCV Model Zoo <https://gluon-cv.mxnet.io/model_zoo/detection.html>`__ and export to the MXNet format using the GluonCV API. For example, for the ``ssd_512_mobilenet1.0`` topology:

   .. code-block:: py
      :force:

      from gluoncv import model_zoo, data, utils
      from gluoncv.utils import export_block
      net = model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
      export_block('ssd_512_mobilenet1.0_voc', net, preprocess=True, layout='HWC')

   As a result, you will get an MXNet model representation in ``ssd_512_mobilenet1.0.params`` and ``ssd_512_mobilenet1.0.json`` files generated in the current directory.

2. Run model conversion API, specifying the ``enable_ssd_gluoncv`` option. Make sure the ``input_shape`` parameter is set to the input shape layout of your model (NHWC or NCHW). The examples below illustrate running model conversion for the SSD and YOLO-v3 models trained with the NHWC layout and located in the ``<model_directory>``:

   * **For GluonCV SSD topologies:**

     .. code-block:: sh

        mo --input_model <model_directory>/ssd_512_mobilenet1.0.params --enable_ssd_gluoncv --input_shape [1,512,512,3] --input data --output_dir <OUTPUT_MODEL_DIR>

   * **For YOLO-v3 topology:**

     * To convert the model:

       .. code-block:: sh

          mo --input_model <model_directory>/yolo3_mobilenet1.0_voc-0000.params  --input_shape [1,255,255,3] --output_dir <OUTPUT_MODEL_DIR>

     * To convert the model with replacing the subgraph with RegionYolo layers:

       .. code-block:: sh

          mo --input_model <model_directory>/models/yolo3_mobilenet1.0_voc-0000.params  --input_shape [1,255,255,3] --transformations_config "front/mxnet/   yolo_v3_mobilenet1_voc.  json" --output_dir <OUTPUT_MODEL_DIR>


