# OpenVINO™ Low Precision Transformations: QuantizationAlignment {#openvino_docs_IE_DG_lpt_QuantizationAlignment}

ngraph::QuantizationAlignmentAttribute class represents the `QuantizationAlignment` attribute.

The attribute defines subgraph with the same quantization alignment. `FakeQuantize` operations are not included. The attribute is used by quantization operations.

| Property name | Values                                       |
|---------------|----------------------------------------------|
| Required      | Yes                                          |
| Defined       | Operation                                    |
| Properties    | value (boolean)                              |