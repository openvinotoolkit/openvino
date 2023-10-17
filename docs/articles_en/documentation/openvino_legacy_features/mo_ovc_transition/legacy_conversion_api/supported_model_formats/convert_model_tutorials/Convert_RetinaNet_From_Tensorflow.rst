.. {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_RetinaNet_From_Tensorflow}

Converting a TensorFlow RetinaNet Model
=======================================


.. meta::
   :description: Learn how to convert a RetinaNet model 
                 from TensorFlow to the OpenVINO Intermediate Representation.


This tutorial explains how to convert a RetinaNet model to the Intermediate Representation (IR).

`Public RetinaNet model <https://github.com/fizyr/keras-retinanet>`__ does not contain pretrained TensorFlow weights.
To convert this model to the TensorFlow format, follow the `Reproduce Keras to TensorFlow Conversion tutorial <https://docs.openvino.ai/2023.1/omz_models_model_retinanet_tf.html>`__. 

After converting the model to TensorFlow format, run the following command:

.. code-block:: sh

   mo --input "input_1[1,1333,1333,3]" --input_model retinanet_resnet50_coco_best_v2.1.0.pb --transformations_config front/tf/retinanet.json


Where ``transformations_config`` command-line parameter specifies the configuration json file containing model conversion hints for model conversion API.
The json file contains some parameters that need to be changed if you train the model yourself. It also contains information on how to match endpoints
to replace the subgraph nodes. After the model is converted to the OpenVINO IR format, the output nodes will be replaced with DetectionOutput layer.

