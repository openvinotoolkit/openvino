## GRUCell <a name="GRUCell"></a> {#openvino_docs_ops_sequence_GRUCell_3}

**Versioned name**: *GRUCell-3*

**Category**: Sequence processing

**Short description**: *GRUCell* represents a single GRU Cell that computes the output using the formula described in the [paper](https://arxiv.org/abs/1406.1078).

**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Required**: *yes*

* *activations*

  * **Description**: activation functions for gates
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid,tanh*
  * **Required**: *no*

* *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* functions attributes
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: None
  * **Required**: *no*

* *clip*

  * **Description**: *clip* specifies value for tensor clipping to be in *[-C, C]* before activations
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: *infinity* that means that the clipping is not applied
  * **Required**: *no*

* *linear_before_reset*

  * **Description**: *linear_before_reset* flag denotes if the layer behaves according to the modification of *GRUCell* described in the formula in the [ONNX documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU).
  * **Range of values**: true or false
  * **Type**: `boolean`
  * **Default value**: false
  * **Required**: *no*

**Inputs**

* **1**: `X` - 2D tensor of type *T* `[batch_size, input_size]`, input data. **Required.**

* **2**: `initial_hidden_state` - 2D tensor of type *T* `[batch_size, hidden_size]`. **Required.**

* **3**: `W` - 2D tensor of type *T* `[3 * hidden_size, input_size]`, the weights for matrix multiplication, gate order: zrh. **Required.**

* **4**: `R` - 2D tensor of type *T* `[3 * hidden_size, hidden_size]`, the recurrence weights for matrix multiplication, gate order: zrh. **Required.**

* **5**: `B` - 1D tensor of type *T*. If *linear_before_reset* is set to 1, then the shape is `[4 * hidden_size]` - the sum of biases for z and r gates (weights and recurrence weights), the biases for h gate are placed separately. Otherwise the shape is `[3 * hidden_size]`, the sum of biases (weights and recurrence weights). **Required.**

**Outputs**

* **1**: `Ho` - 2D tensor of type *T* `[batch_size, hidden_size]`, the last output value of hidden state.

**Types**

* *T*: any supported floating-point type.

**Example**
```xml
<layer ... type="GRUCell" ...>
    <data hidden_size="128" linear_before_reset="1"/>
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
            <dim>384</dim>
            <dim>16</dim>
        </port>
         <port id="3">
            <dim>384</dim>
            <dim>128</dim>
        </port>
         <port id="4">
            <dim>768</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>1</dim>
            <dim>128</dim>
        </port>
    </output>
</layer>
```
