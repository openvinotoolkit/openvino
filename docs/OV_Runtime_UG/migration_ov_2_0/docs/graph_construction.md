# OpenVINO™ graph construction {#openvino_graph_construction}

OpenVINO™ 2.0 includes nGraph engine in a common part. The `ngraph` namespace was changed to `ov`.
Code snippets below show how application code should be changed for migration to OpenVINO™ 2.0.

nGraph API:

@snippet snippets/ngraph.cpp ngraph:graph

OpenVINO™ 2.0 API:

@snippet snippets/ov_graph.cpp ov:graph
