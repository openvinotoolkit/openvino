# OpenVINO™ Low Precision Transformations: IntervalsAlignment {#openvino_docs_IE_DG_lpt_IntervalsAlignment}

The attribute defines subgraph with the same quantization intervals alignment. `FakeQuantize` operations are included. The attribute is used by quantization operations.

| Property name | Values                                       |
|---------------|----------------------------------------------|
| Required      | Yes                                          |
| Defined       | Operation                                    |
| Properties    | combined interval, minimal interval, minimal levels, preferable precisions |