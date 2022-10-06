# How to support a new OpenVINO opset

Adding a new opset to the OpenVINO IR Frontend is a very frequent task. When you introduce a new operation set, you must also support it inside the Frontend.

To do this, you need to add a new line to the constructor of the `InputModelIRImpl` class:

https://github.com/openvinotoolkit/openvino/blob/5c1ddd32de2ad4f9ec0c2dbc8b256add10896ec3/src/frontends/ir/src/input_model.cpp#L199-L218

This line adds a new opset to the `m_opsets` map.

## See also

 * [OpenVINO IR Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
