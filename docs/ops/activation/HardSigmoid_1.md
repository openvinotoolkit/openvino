## HardSigmoid <a name="HardSigmoid"></a> {#openvino_docs_ops_activation_HardSigmoid_1}

**Versioned name**: *HardSigmoid-1*

**Category**: *Activation function*

**Short description**: HardSigmoid element-wise activation function.

**Attributes**: *HardSigmoid* operation has no attributes.

**Mathematical Formulation**

For each element from the input tensor calculates corresponding
 element in the output tensor with the following formula:
 \f[
 y = max(0, min(1, alpha * x + beta))
 \f]

**Inputs**

* **1**: An tensor of type *T*. **Required.**

* **2**: `alpha` 0D tensor (scalar) of type *T*. **Required.**

* **3**: `beta` 0D tensor (scalar) of type *T*. **Required.**

**Outputs**

* **1**: The result of the hard sigmoid operation. A tensor of type *T*.

**Types**

* *T*: any floating-point type.

**Examples**

```xml
<layer ... type="HardSigmoid">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
        <port id="1"/>
        <port id="2"/>
    </input>
    <output>
        <port id="3">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```
