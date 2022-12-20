# 在 OpenVINO™ 运行时创建模型 {#openvino_2_0_model_creation_zh_CN}

采用 API 2.0 的 OpenVINO™ 运行时将 nGraph 引擎作为通用部件。`ngraph` 命名空间已更改为 `ov`，但保留了 nGraph API 的所有其他部件。

以下代码片段显示了如何更改应用代码以便迁移到 API 2.0。

## nGraph API

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ngraph.cpp ngraph:graph
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ngraph.py ngraph:graph
@endsphinxtab

@endsphinxtabset

## API 2.0

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_graph.cpp ov:graph
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_graph.py ov:graph
@endsphinxtab

@endsphinxtabset

## 其他资源

- [Hello 模型创建 C++ 示例](../../../../samples/cpp/model_creation_sample/README.md)
- [Hello 模型创建 Python 示例](../../../../samples/python/model_creation_sample/README.md)
