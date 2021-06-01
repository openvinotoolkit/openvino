## AdaptiveMaxPool<a name="AdaptiveMaxPool"></a> {#openvino_docs_ops_pooling_AdaptiveMaxPool_8}

**Versioned name**: *AdaptiveMaxPool-8*

**Category**: *Pooling*

**Short description**: Applies max pooling with adaptive kernel size over the input.

**Detailed description**: This operation calculates the output based on the 1st input and `output_size` determined by the 2nd input.
The kernel dimensions are calculated using the following formulae for the `NCDHW` input case:

\f[
\begin{array}{lcl}
d_{start} &=& floor(i*D_{in}/D_{out})\\
d_{end}   &=& ceil((i+1)*D_{in}/D_{out})\\
h_{start} &=& floor(j*H_{in}/H_{out})\\
h_{end}   &=& ceil((j+1)*H_{in}/H_{out})\\
w_{start} &=& floor(k*W_{in}/W_{out})\\
w_{end}   &=& ceil((k+1)*W_{in}/W_{out})
\end{array}
\f]

The output is calculated following this formula:

\f[
Output(i,j,k) = max(Input[d_{start}:d_{end}, h_{start}:h_{end}, w_{start}:w_{end}])
\f]

**Attributes**:

*   *output_type*

  * **Description**: the type of the 2nd output containing indices
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *No*

**Inputs**:

*   **1**: 3D, 4D or 5D input tensor of shape `[N,C,L]`, `[N,C,H,W]` or `[N,C,D,H,W]` and type *T*. Required.
*   **2**: 1D tensor describing output shape for spatial dimensions. Can be `[L_out]` for 3D input, `[H_out,W_out]` for 4D input, `[D_out,H_out,W_out]` for 5D input and of type *T_IND*. Required.

**Outputs**:

*   **1**: Output of type *T* and shape `[N,C,L_out]`, `[N,C,H_out,W_out]` or `[N,C,D_out,H_out,W_out]`.
*   **2**: Output of type specified by *output_type* and same shape as 1st output containing indices of elements in 1st output. The values of indices are computed as if input was flatten 1-D tensor.

**Types**

*   *T*: floating-point type.
*   *T_IND*: `int32` or `int64`.

**Examples**

```xml
<layer ... type="AdaptiveMaxPool" ... >
    <data output_type="i64"/>
    <input> 
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>32</dim>
            <dim>32</dim>
        </port>
    </input>
    <input> 
        <port id="1">
            <dim>2</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>3</dim>
            <dim>16</dim>
            <dim>16</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>3</dim>
            <dim>16</dim>
            <dim>16</dim>
        </port>
    </output>
</layer>
```