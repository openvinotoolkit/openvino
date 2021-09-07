## LSTMCell <a name="LSTMCell"></a> {#openvino_docs_ops_sequence_LSTMCell_1}

**Versioned name**: *LSTMCell-1*

**Category**: *Sequence processing*

**Short description**: *LSTMCell* operation represents a single LSTM cell. It computes the output using the formula described in the original paper [Long Short-Term Memory](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf).

**Detailed description**

```
Formula:
  *  - matrix mult
 (.) - eltwise mult
 [,] - concatenation
sigm - 1/(1 + e^{-x})
tanh - (e^{2x} - 1)/(e^{2x} + 1)
   f = sigm(Wf*[Hi, X] + Bf)
   i = sigm(Wi*[Hi, X] + Bi)
   c = tanh(Wc*[Hi, X] + Bc)
   o = sigm(Wo*[Hi, X] + Bo)
  Co = f (.) Ci + i (.) c
  Ho = o (.) tanh(Co)
```

**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Required**: *yes*

* *activations*

  * **Description**: *activations* specifies activation functions for gates, there are three gates, so three activation functions should be specified as a value for this attributes
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid,tanh,tanh*
  * **Required**: *no*

* *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* attributes of functions; applicability and meaning of these attributes depends on chosen activation functions
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: None
  * **Required**: *no*

* *clip*

  * **Description**: *clip* specifies bound values *[-C, C]* for tensor clipping. Clipping is performed before activations.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: *infinity* that means that the clipping is not applied
  * **Required**: *no*

**Inputs**

* **1**: `X` - 2D tensor of type *T* `[batch_size, input_size]`, input data. **Required.**

* **2**: `initial_hidden_state` - 2D tensor of type *T* `[batch_size, hidden_size]`. **Required.**

* **3**: `initial_cell_state` - 2D tensor of type *T* `[batch_size, hidden_size]`. **Required.**

* **4**: `W` - 2D tensor of type *T* `[4 * hidden_size, input_size]`, the weights for matrix multiplication, gate order: fico. **Required.**

* **5**: `R` - 2D tensor of type *T* `[4 * hidden_size, hidden_size]`, the recurrence weights for matrix multiplication, gate order: fico. **Required.**

* **6**: `B` 1D tensor of type *T* `[4 * hidden_size]`, the sum of biases (weights and recurrence weights). **Required.**


**Outputs**

* **1**: `Ho` - 2D tensor of type *T* `[batch_size, hidden_size]`, the last output value of hidden state.

* **2**: `Co` - 2D tensor of type *T* `[batch_size, hidden_size]`, the last output value of cell state.

**Types**

* *T*: any supported floating-point type.

**Example**
```xml
<layer ... type="LSTMCell" ...>
    <data hidden_size="128"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>16</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>128</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>128</dim>
        </port>
         <port id="3">
            <dim>512</dim>
            <dim>16</dim>
        </port>
         <port id="4">
            <dim>512</dim>
            <dim>128</dim>
        </port>
         <port id="5">
            <dim>512</dim>
        </port>
    </input>
    <output>
        <port id="6">
            <dim>1</dim>
            <dim>128</dim>
        </port>
        <port id="7">
            <dim>1</dim>
            <dim>128</dim>
        </port>
    </output>
</layer>
```
