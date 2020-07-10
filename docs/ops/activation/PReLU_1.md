## PReLU <a name="PReLU"></a> {#openvino_docs_ops_activation_PReLU_1}

**Versioned name**: *PReLU-1*

**Category**: Activation function

**Short description**: *PReLU* performs element-wise parametric ReLU operation with negative slope defined by the second input.

**Attributes**: operation has no attributes.

**Inputs**

* **1**: `X` - Input tensor of any supported floating point type T1. Required.

* **2**: `slope` - Tensor with negative slope values of type T2. The shape of the tensor should be broadcastable to input 1. Required.

**Outputs**

* **1**: The result of element-wise PReLU operation applied for tensor from input 1 with slope values from input 2. A tensor of type T1 and shape matching shape of input *x* tensor.

**Types**

* *T1*: arbitrary supported floating point type.

* *T2*: arbitrary supported floating point type.

**Detailed description**
Before performing addition operation, input tensor 2 with slope values is broadcasted to input 1.
The broadcasting rules are aligned with ONNX Broadcasting. Description is available in <a href="https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md">ONNX docs</a>.

After broadcasting *PReLU* does the following for each input 1 element x:

    f(x) = slope * x for x < 0; x for x >= 0