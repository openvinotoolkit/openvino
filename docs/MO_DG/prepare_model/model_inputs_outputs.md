# Model Inputs and Outputs, Shapes and Layouts {#openvino_docs_model_inputs_outputs}

@sphinxdirective

Users interact with a model by passing data to its *inputs* before the inference and retrieving data from its *outputs* after the inference. A model may have one or multiple inputs and outputs. Normally, in OpenVINO™ toolkit, all inputs and outputs in the converted model are identified in the same way as in the original framework model.

OpenVINO uses the *names of tensors* for identification. Depending on the framework, the names of tensors are formed differently.

A model accepts inputs and produces outputs of some *shape*. Shape defines the number of dimensions in a tensor and their order. For example, an image classification model can accept tensor of shape [1, 3, 240, 240] and produces tensor of shape [1, 1000].

The meaning of each dimension in the shape is specified by its *layout*. Layout is an interpretation of shape dimensions. OpenVINO toolkit conversion tools and APIs keep all dimensions and their order unchanged and aligned with the original framework model. Usually, original models do not contain layout information explicitly, but in various pre-processing and post-processing scenarios in the OpenVINO Runtime API, sometimes it is required to have the layout specified explicitly. We recommend specifying layouts for inputs/outputs during the model conversion.

OpenVINO also supports *partially defined shapes*, where part of the dimensions is undefined. Undefined dimensions are also kept intact in the final IR file and you can  define them later, during runtime. Undefined dimensions can be used as :doc:`dynamic dimensions <openvino_docs_OV_UG_DynamicShapes>` for certain hardware and models, which enables you to change shapes of input data dynamically in each infer request. For example, the sequence length dimension in the BERT model can be left undefined and variously sized data along this dimension can be fed on the CPU.

To learn about how the model is represented in OpenVINO™ Runtime, see the :doc:`Model Representation in OpenVINO™ Runtime <openvino_docs_OV_UG_Model_Representation>`.

@endsphinxdirective