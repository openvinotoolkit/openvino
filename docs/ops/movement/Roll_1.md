## Roll <a name="Roll"></a> {#openvino_docs_ops_movement_Roll_1}

**Versioned name**: *Roll-1*

**Category**: data movement operation

**Short description**: *Roll* operation shifts elements of a tensor along specified axis.

**Detailed description**: *Roll* produces a tensor with the same shape as the first input tensor and with elements shifted along dimensions specified in *axis* tensor. The shift size is specified in *shift* input tensor. If the value of *shift* is positive, elements are shifted positively (towards larger indices). Otherwise elements are shifted negatively (towards smaller indices). Elements that are shifted beyond the last position are added in the same order starting from the first position.

**Attributes**

No attributes available.

**Inputs**:

*   **1**: `data` the tensor of type *T1*. **Required.**

*   **2**: `shift` scalar or 1D tensor of type *T2* which specifies the number of places by which the elements of `data` tensor are shifted. If `shift` is a scalar, each dimension specified in `axis` tensor are rolled by same `shift` value. If `shift` is 1D tensor, `axis` must be a 1D tensor of the same size, and each dimension from `axis` tensor are rolled by the corresponding value from `shift` tensor. **Required.**

*   **3**: `axis` scalar or 1D tensor of type *T2* which specifies axes along which elements are shifted. If the `axis` is not specified, `data` tensor is flattened before shifting and then restored to the original shape. If the same axis is referenced more than once, the total shift for that axis will be the sum of all the shifts that belong to that axis.


**Outputs**:

*   **1**: output tensor with shape and type equal to `data` tensor.

**Types**

* *T1*: any supported type.
* *T2*: any supported integer type.

**Example**

*Example 1: "shift" and "axis" are 1-D tensors.*

```xml
<layer ... type="Roll">
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
        <port id="1">
            <dim>2</dim>
        </port>
        <port id="2">
            <dim>2</dim> <!-- shifting along specified axes with the corresponding shift values -->
        </port>
     </input>
    <output>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
    </output>
</layer>
```

*Example 2: "shift" value is a scalar and multiple axes are specified.*

```xml
<layer ... type="Roll">
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
        <port id="1">
            <dim>1</dim>
        </port>
        <port id="2">
            <dim>2</dim> <!-- shifting along specified axes with the same shift value -->
        </port>
     </input>
    <output>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
    </output>
</layer>
```

*Example 3: axis is not specified.*

```xml
<layer ... type="Roll">
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
        <port id="1">
            <dim>1</dim>   <!-- shifting along single axis of the flattened data tensor, as axis is not specified -->
        </port>
     </input>
    <output>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
    </output>
</layer>
```
