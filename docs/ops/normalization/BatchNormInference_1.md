## BatchNormInference <a name="BatchNormInference"></a> {#openvino_docs_ops_normalization_BatchNormInference_1}

**Versioned name**: *BatchNormInference-1*

**Category**: *Normalization*

**Short description**: *BatchNormInference* layer normalizes a `input` tensor by `mean` and `variance`, and applies a scale (`gamma`) to it, as well as an offset (`beta`).

**Attributes**:

* *epsilon*
  * **Description**: *epsilon* is the number to be added to the variance to avoid division by zero when normalizing a value. For example, *epsilon* equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: `input` - input tensor with data for normalization. At least a 2D tensor of type T, the second dimension represents the channel axis and must have a span of at least 1. **Required.**
* **2**: `gamma` - gamma scaling for normalized value. A 1D tensor of type T with the same span as input's channel axis. **Required.**
* **3**: `beta` - bias added to the scaled normalized value. A 1D tensor of type T with the same span as input's channel axis.. **Required.**
* **4**: `mean` - value for mean normalization. A 1D tensor of type T with the same span as input's channel axis.. **Required.**
* **5**: `variance` - value for variance normalization. A 1D tensor of type T with the same span as input's channel axis.. **Required.**

**Outputs**

* **1**: The result of normalization. A tensor of the same type and shape with 1st input tensor.

**Types**

* *T*: any numeric type.

**Mathematical Formulation**

*BatchNormInference*  normalizes the output in each hidden layer.
*   **Input**: Values of \f$x\f$ over a mini-batch:
    \f[
    \beta = \{ x_{1...m} \}
    \f]
*   **Parameters to learn**: \f$ \gamma, \beta\f$
*   **Output**:
    \f[
    \{ o_{i} = BN_{\gamma, \beta} ( b_{i} ) \}
    \f]
*   **Mini-batch mean**:
    \f[
    \mu_{\beta} \leftarrow \frac{1}{m}\sum_{i=1}^{m}b_{i}
    \f]
*   **Mini-batch variance**:
    \f[
    \sigma_{\beta }^{2}\leftarrow \frac{1}{m}\sum_{i=1}^{m} ( b_{i} - \mu_{\beta} )^{2}
    \f]
*   **Normalize**:
    \f[
    \hat{b_{i}} \leftarrow \frac{b_{i} - \mu_{\beta}}{\sqrt{\sigma_{\beta }^{2} + \epsilon }}
    \f]
*   **Scale and shift**:
    \f[
    o_{i} \leftarrow \gamma\hat{b_{i}} + \beta = BN_{\gamma ,\beta } ( b_{i} )
    \f]

**Example**

```xml
<layer ... type="BatchNormInference" ...>
    <data epsilon="9.99e-06" />
    <input>
        <port id="0">  <!-- input -->
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
        <port id="1">  <!-- gamma -->
            <dim>3</dim>
        </port>
        <port id="2">  <!-- beta -->
            <dim>3</dim>
        </port>
        <port id="3">  <!-- mean -->
            <dim>3</dim>
        </port>
        <port id="4">  <!-- variance -->
            <dim>3</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </output>
</layer>
```
