## AdaptiveAvgPool<a name="AdaptiveAvgPool"></a> {#openvino_docs_ops_pooling_AdaptiveAvgPool_8}

**Versioned name**: *AdaptiveAvgPool-8*

**Category**: *Pooling*

**Short description**: Applies average pooling with adaptive kernel size over the input.

**Detailed description**: This operation calculates the output based on the first input and `output_size` determined by the second input.
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

The output is calculated with the following formula:

\f[
Output(i,j,k) = \frac{Input[d_{start}:d_{end}, h_{start}:h_{end}, w_{start}:w_{end}]}{(d_{end}-d_{start})*(h_{end}-h_{start})*(w_{end}-w_{start})}
\f]

**Inputs**:

*   **1**: 3D, 4D, or 5D input tensor of shape `[N, C, H]`, `[N, C, H, W]` or `[N, C, D, H, W]` and type *T*. **Required.**
*   **2**: 1D tensor describing output shape for spatial dimensions. Can be `[H_out]` for 3D input, `[H_out, W_out]` for 4D input, `[D_out, H_out, W_out]` for 5D input and of type *T_SHAPE*. **Required.**

**Outputs**:

*   **1**: Output of type *T* and shape `[N, C, H_out]`, `[N, C, H_out, W_out]` or `[N, C, D_out, H_out, W_out]`.

**Types**

*   *T*: floating-point type.
*   *T_SHAPE*: `int32` or `int64`.

**Examples**

```xml
<layer ... type="AdaptiveAvgPool" ... >
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
        <port id="2">
            <dim>1</dim>
            <dim>3</dim>
            <dim>16</dim>
            <dim>16</dim>
        </port>
    </output>
</layer>
```
