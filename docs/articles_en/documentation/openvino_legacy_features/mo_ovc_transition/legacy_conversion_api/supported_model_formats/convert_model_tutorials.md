# Model Conversion Tutorials {#openvino_docs_MO_DG_prepare_model_convert_model_tutorials}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_AttentionOCR_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_BERT_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_CRNN_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_DeepSpeech_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_EfficientDet_Models
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_FaceNet_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_GNMT_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_lm_1b_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_NCF_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_RetinaNet_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Slim_Library_Models
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_WideAndDeep_Family_Models
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_XLNet_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow
   openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_Faster_RCNN
   openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_GPT2
   openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_Mask_RCNN
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_Bert_ner
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_Cascade_RCNN_res101
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_F3Net
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_QuartzNet
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_RCAN
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_RNNT
   openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_YOLACT


.. meta::
   :description: Get to know conversion methods for specific TensorFlow, ONNX, PyTorch, MXNet, and Kaldi models.


This section provides a set of tutorials that demonstrate conversion methods for specific 
TensorFlow, ONNX, and PyTorch models. Note that these instructions do not cover all use 
cases and may not reflect your particular needs.
Before studying the tutorials, try to convert the model out-of-the-box by specifying only the 
``--input_model`` parameter in the command line.

.. note::

   Apache MXNet, Caffe, and Kaldi are no longer directly supported by OpenVINO. 
   They will remain available for some time, so make sure to transition to other 
   frameworks before they are fully discontinued.
   
You will find a collection of :doc:`Python tutorials <tutorials>` written for running on Jupyter notebooks 
that provide an introduction to the OpenVINO™ toolkit and explain how to use the Python API and tools for 
optimized deep learning inference.

@endsphinxdirective