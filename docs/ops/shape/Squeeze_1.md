## Squeeze<a name="Squeeze"></a> {#openvino_docs_ops_shape_Squeeze_1}

**Versioned name**: *Squeeze-1*

**Category**: Shape manipulation

**Short description**: *Squeeze* removes dimensions equal to 1 from the first input tensor.

**Detailed description**: *Squeeze* can be used with or without the second input tensor.
* If only the first input is provided, every dimension that is equal to 1 will be removed from it.
* With the second input provided, each value is an index of a dimension from the first tensor that is to be removed. Specified dimension has to be equal to 1, otherwise an error will be raised. Dimension indices can be specified directly, or by negative indices (counting dimensions from the end).

**Attributes**: *Squeeze* operation doesn't have attributes.

**Inputs**:

*   **1**: Multidimensional input tensor of type *T*. **Required.**

*   **2**: Scalar or 1D tensor of type *T_INT* with indices of dimensions to squeeze. Values could be negative (have to be from range `[-R, R-1]`, where `R` is the rank of the first input). **Optional.**

**Outputs**:

*   **1**: Tensor with squeezed values of type *T*.

**Types**

* *T*: any numeric type.

* *T_INT*: any supported integer type.

**Example**

*Example 1: squeeze 4D tensor to a 2D tensor*
```xml
<layer ... type="Squeeze">
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>1</dim>
            <dim>2</dim>
        </port>
    </input>
    <input>
        <port id="1">
            <dim>2</dim>  <!-- value [0, 2] -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```

*Example 2: squeeze 1D tensor with 1 element to a 0D tensor (constant)*
```xml
<layer ... type="Squeeze">
    <input>
        <port id="0">
            <dim>1</dim>
        </port>
    </input>
    <input>
        <port id="1">
            <dim>1</dim>  <!-- value is [0] -->
        </port>
    </input>
    <output>
        <port id="2">
        </port>
    </output>
</layer>
```
