## ConvolutionBackpropData<a name="ConvolutionBackpropData"></a> {#openvino_docs_ops_convolution_ConvolutionBackpropData_1}

**Versioned name**: *ConvolutionBackpropData-1*

**Category**: Convolution

**Short description**: Computes the gradients of a Convolution operation with respect to the input. Also known as a Deconvolution or a Transposed Convolution.

**Detailed description**:

ConvolutionBackpropData takes the input tensor, weights tensor and output shape and computes the output tensor of a given shape. The shape of the output can be specified as an input 1D integer tensor explicitly or determined by other attributes implicitly. If output shape is specified as an explicit input, shape of the output exactly matches the specified size and required amount of padding is computed.

ConvolutionBackpropData accepts the same set of attributes as a regular Convolution operation, but they are interpreted in a "backward way", so they are applied to the output of ConvolutionBackpropData, but not to the input. Refer to a regular Convolution operation for detailed description of each attribute.

Output shape when specified as an input `output_shape`, specifies only spatial dimensions. No batch or channel dimension should be passed along with H, W or other spatial dimensions. If `output_shape` is omitted, then `pads_begin`, `pads_end` or `auto_pad` are used to determine output spatial shape `[Y_1, Y_2, ..., Y_D]` by input spatial shape `[X_1, X_2, ..., X_D]` in the following way:

```
if auto_pads != None:
    pads_begin[i] = 0
    pads_end[i] = 0

Y_i = stride[i] * (X_i - 1) + ((K_i - 1) * dilations[i] + 1) - pads_begin[i] - pads_end[i] + output_padding[i]
```

where `K_i` filter kernel dimension along spatial axis `i`.

 If `output_shape` is specified, `pads_begin` and `pads_end` are ignored, and `auto_pad` defines how to distribute padding amount around the tensor. In this case pads are determined based on the next formulas to correctly align input and output tensors (similar to ONNX definition at https://github.com/onnx/onnx/blob/master/docs/Operators.md#convtranspose):

```
total_padding[i] = stride[i] * (X_i - 1) + ((K_i - 1) * dilations[i] + 1) - output_shape[i] + output_padding[i]
if auto_pads != SAME_UPPER:
    pads_begin[i] = total_padding[i] // 2
    pads_end[i] = total_padding[i] - pads_begin[i]
else:
    pads_end[i] = total_padding[i] // 2
    pads_begin[i] = total_padding[i] - pads_end[i]
```

**Attributes**

* *strides*

  * **Description**: *strides* has the same definition as *strides* for a regular Convolution but applied in the backward way, for the output tensor.
  * **Range of values**: positive integers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* has the same definition as *pads_begin* for a regular Convolution but applied in the backward way, for the output tensor. May be omitted specified, in which case pads are calculated automatically.
  * **Range of values**: non-negative integers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* has the same definition as *pads_end* for a regular Convolution but applied in the backward way, for the output tensor. May be omitted, in which case pads are calculated automatically.
  * **Range of values**: non-negative integers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.
  
* *dilations*

  * **Description**: *dilations* has the same definition as *dilations* for a regular Convolution but applied in the backward way, for the output tensor.
  * **Range of values**: positive integers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* has the same definition as *auto_pad* for a regular Convolution but applied in the backward way, for the output tensor. 
    * *explicit*: use explicit padding values from `pads_begin` and `pads_end`.
    * *same_upper (same_lower)* the input is padded to match the output size. In case of odd padding value an extra padding is added at the end (at the beginning).
    * *valid* - do not use padding.
  * **Type**: string
  * **Default value**: None
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

* *output_padding*

  * **Description**: *output_padding* adds additional amount of paddings per each spatial axis in the `output` tensor. It unlocks more elements in the output allowing them to be computed. Elements are added at the higher coordinate indices for the spatial dimensions. Number of elements in *output_padding* list matches the number of spatial dimensions in `data` and `output` tensors.
  * **Range of values**: non-negative integer values
  * **Type**: int[]
  * **Default value**: all zeros
  * **Required**: *no*

**Inputs**:

*   **1**: `data` -- input tensor of rank 3 or greater. Layout is `[N, C_INPUT, X1, ..., XD]`. *Required*.

*   **2**: `filter` -- convolution kernel tensor. Weights have shape `[C_INPUT, C_OUTPUT, K_D, ..., K_1]`. `C_INPUT` is the number of channels in input `data` tensor shape, and `C_OUTPUT` is the number of channels in the `output` tensor. Spatial size of the kernel `[K_D, ..., K_1]` is derived from the shape of this input and aren't specified by any attribute. *Required*.

*   **3**: `output_shape` is 1D integer tensor that specifies spatial shape of the output. *Optional*. If specified, *padding amount* is deduced from relation of input and output spatial shapes according to formulas in the description. If not specified, *output shape* is calculated based on the `pads_begin` and `pads_end` or completely according to `auto_pad`.

**Outputs**:

*   **1**: `output` -- output tensor of the same rank as input `data` tensor and shape `[N, C_OUTPUT, Y1, ..., YD]`.

**Example**

```xml
<layer id="5" name="upsampling_node" type="ConvolutionBackpropData">
    <data dilations="1,1" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>20</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
        <port id="1">
            <dim>20</dim>
            <dim>10</dim>
            <dim>3</dim>
            <dim>3</dim>
        </port>
    </input>
    <output>
        <port id="0" precision="FP32">
            <dim>1</dim>
            <dim>10</dim>
            <dim>447</dim>
            <dim>447</dim>
        </port>
    </output>
</layer>
```
