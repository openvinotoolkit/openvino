## Swish <a name="Swish"></a>

**Versioned name**: *Swish-4*

**Category**: *Activation*

**Short description**: Swish takes one input tensor and produces output tensor where the Swish function is applied to the tensor elementwise.

**Detailed description**: For each element from the input tensor calculates corresponding
element in the output tensor with the following formula: 

    \f[
    Swish(x) = x / (1.0 + e^{-(beta * x)})
    \f]

The Swish operation is introduced in the [article](https://arxiv.org/pdf/1710.05941.pdf).

**Attributes**:

**Inputs**:

*   **1**: Multidimensional input tensor of type *T*. **Required**.

*   **2**: Scalar with non-negative value of type *T*. Multiplication parameter *beta* for the sigmoid. If the input is not connected then the default value 1.0 is used. **Optional**

**Outputs**:

*   **1**: The resulting tensor of the same shape and type as input tensor.

**Types**

* *T*: arbitrary supported floating point type.


**Example**

```xml
<layer ... type="Swish">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
        <port id="1"/>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```