# Model creation {#openvino_2_0_model_creation}

OpenVINO™ Runtime API 2.0 includes nGraph engine as a common part. The `ngraph` namespace was changed to `ov`, all other ngraph API is preserved as is.
Code snippets below show how application code should be changed for migration to OpenVINO™ Runtime API 2.0.

nGraph API:

@snippet snippets/ngraph.cpp ngraph:graph

OpenVINO™ Runtime API 2.0:

@snippet snippets/ov_graph.cpp ov:graph

See also:
[Hello Model Creation C++ Sample](../../../samples/cpp/model_creation_sample/README.md)
[Hello Model Creation Python Sample](../../../samples/python/model_creation_sample/README.md)
