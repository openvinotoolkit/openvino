## SoftPlus <a name="SoftPlus"></a>

**Versioned name**: *SoftPlus-4*

**Category**: *Activation*

**Short description**: SoftPlus takes one input tensor and produces output tensor where the softplus function is applied to the tensor elementwise.

**Detailed description**: For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:

    \f[
    SoftPlus(x) = ln(e^{x} + 1.0)
    \f]

**Attributes**: *SoftPlus* operation has no attributes.


**Inputs**:

*   **1**: Multidimensional input tensor of type *T*. **Required**.

**Outputs**:

*   **1**: The resulting tensor of the same shape and type as input tensor.

**Types**

* *T*: arbitrary supported floating point type.


**Example**

```xml
<layer ... type="SoftPlus">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```