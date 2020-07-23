## Squeeze<a name="Squeeze"></a> {#openvino_docs_ops_shape_Squeeze_1}

**Versioned name**: *Squeeze-1*

**Category**: Shape manipulation

**Short description**: *Squeeze* removes specified dimensions (second input) equal to 1 of the first input tensor. If the second input is omitted then all dimensions equal to 1 are removed. If the specified dimension is not equal to one then error is raised.

**Attributes**: *Squeeze* operation doesn't have attributes.

**Inputs**:

*   **1**: Multidimensional input tensor. Required.

*   **2**: `(optional)`: 0D or 1D tensor with dimensions indices to squeeze. Values could be negative. Indices could be integer or float values.

**Example**

*Example 1:*
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