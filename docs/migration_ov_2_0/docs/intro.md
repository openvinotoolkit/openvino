# OpenVINO™ API 2.0 Transition Guide {#openvino_2_0_transition_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_inference_pipeline
   openvino_graph_construction
      
@endsphinxdirective

The OpenVINO™ API 2.0 introduced in order to simplify migration from other frameworks and make the OpenVINO™ API more user-friendly.
The list with differences between APIs below:

 - OpenVINO™ API 2.0 uses tensor names or indexes to work with Inputs or Outputs, the old API works with operation names.
 - Structures for Shapes, element types were changed.
 - Naming style was changed. The old API uses CamelCaseStyle and OpenVINO™ API 2.0 uses snake_case for function names.
 - Namespaces were aligned between components.

Please look at next transition guides to understand how transit own application to OpenVINO™ API 2.0.
 - [OpenVINO™ Graph Construction](graph_construction.md)
 - [OpenVINO™ Common Inference pipeline](common_inference_pipeline.md)
