# OpenVINO™ API 2.0 transition guide {#ov_2_0_transition_guide}

The OpenVINO™ API 2.0 introduced in order to simplify migration from other frameworks and make the OpenVINO™ API more user-friendly.
The list with differences between APIs below:

 - OpenVINO™ API 2.0 uses tensor names or indexes to work with Inputs or Outputs, the old API works with operation names.
 - Structures for Shapes, element types were changed.
 - Naming style was changed. The old API uses CamelCaseStyle and OpenVINO™ API 2.0 uses snake_case for function names.
 - Namespaces were aligned between components.

Please look at next transition guides to understand how transit own application to OpenVINO™ API 2.0.
 - [OpenVINO™ Common Inference pipeline](@ref ov_inference_pipeline)
