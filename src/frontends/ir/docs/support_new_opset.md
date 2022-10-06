# How to support new OpenVINO opset

The addition of new opset to OpenVINO IR frontend is a very often task, each time when we introduce new operation set we also need to support it inside the Frontend.

You need to add only new line in to the constructor of `InputModelIRImpl` class:

https://github.com/openvinotoolkit/openvino/blob/5c1ddd32de2ad4f9ec0c2dbc8b256add10896ec3/src/frontends/ir/src/input_model.cpp#L199-L218

This line should add new opset to `m_opsets` map.

## See also

 * [OpenVINO IR Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
