## ReLU <a name="ReLU"></a>

**Versioned name**: *ReLU-1*

**Category**: *Activation*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/relu.html)

**Detailed description**: [Reference](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#rectified-linear-units)

**Attributes**: *ReLU* operation has no attributes.

**Mathematical Formulation**

\f[
Y_{i}^{( l )} = max(0, Y_{i}^{( l - 1 )})
\f]

**Inputs**:

*   **1**: Multidimensional input tensor. Required.

**Example**

```xml
<layer ... type="ReLU">
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