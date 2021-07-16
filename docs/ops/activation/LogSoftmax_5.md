## LogSoftMax <a name="LogSoftmax"></a> {#openvino_docs_ops_activation_LogSoftmax_5}

**Versioned name**: *LogSoftmax-5*

**Category**: *Activation*

**Short description**: LogSoftmax computes the natural logarithm of softmax values for the given input.

**Note**:  This is recommended to not compute LogSoftmax directly as Log(Softmax(x, axis)), more numeric stable is to compute LogSoftmax as:
\f[
t = (x - ReduceMax(x, axis)) \\
LogSoftmax(x, axis) = t - Log(ReduceSum(Exp(t), axis))
\f]

**Attributes**

* *axis*

  * **Description**: *axis* represents the axis of which the *LogSoftmax* is calculated. Negative value means counting dimensions from the back.
  * **Range of values**: any integer value
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: Input tensor *x* of type *T* with enough number of dimension to be compatible with *axis* attribute. **Required.**

**Outputs**:

*   **1**: The resulting tensor of the same shape and of type *T*.

**Types**

* *T*: any floating-point type.

**Mathematical Formulation**

\f[
y_{c} = ln\left(\frac{e^{Z_{c}}}{\sum_{d=1}^{C}e^{Z_{d}}}\right)
\f]
where \f$C\f$ is a size of tensor along *axis* dimension.

**Example**

```xml
<layer ... type="LogSoftmax" ... >
    <data axis="1" />
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="3">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```
