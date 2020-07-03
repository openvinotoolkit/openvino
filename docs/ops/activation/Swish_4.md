## Swish <a name="Swish"></a>

**Versioned name**: *Swish-4*

**Category**: *Activation*

**Short description**: Swish takes one input tensor and produces output tensor where the swish function is applied to the tensor elementwise.

**Detailed description**: [Reference.](https://arxiv.org/pdf/1710.05941.pdf)    
For each element from the input tensor calculates corresponding
element in the output tensor with the following formula: `x * sigmoid(beta * x)`

**Attributes**:

* *beta*

  * **Description**: *beta* multiplication parameter for sigmoid.
  * **Range of values**: non-negative positive floating point number
  * **Type**: float
  * **Default value**: 1.0
  * **Required**: *no*


**Inputs**:

*   **1**: Multidimensional input tensor of type *T*. **Required**.

**Outputs**:

*   **1**: The resulting tensor of the same shape and type as input tensor.

**Types**

* *T*: any supported type.


**Example**

```xml
<layer ... type="Swish">
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