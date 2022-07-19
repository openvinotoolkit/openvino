# AUGRUSequence  {#openvino_docs_ops_sequence_AUGRUSequence_10}

**Versioned name**: *AUGRUSequence-10*

**Category**: *Sequence processing*

**Short description**: *AUGRUSequence* operation represents a series of AUGRU cells (GRU with attentional update gate).

**Detailed description**: The difference between *AUGRUSequence* and [GRUSequence](./GRUSequence_5.md) is the additional attention score input `A`, which is a multiplier for the update gate.
The AUGRU formula is based on the [paper arXiv:1809.03672](https://arxiv.org/abs/1809.03672).

The sequence can be connected differently depending on `direction` attribute that specifies the direction of traversing of input data along sequence dimension or specifies whether it should be a bidirectional sequence.

All of the attributes are in sync with the specification of [GRUSequence](./GRUSequence_5.md).


```
AUGRU formula:
  *  - matrix multiplication
 (.) - Hadamard product (element-wise)

 f, g - activation functions
 z - update gate, r - reset gate, h - hidden gate
 a - attention score

  rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)  # default, when 'linear_before_reset' is set False
  ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)  # when 'linear_before_reset' is set True

  ####### TO BE DECIDED #######
  # Version I
  zt' = at (.) zt  # multiplication by attention score

  # Version II
  zt' = 1 - (at (.) zt)  # multiplication by attention score

  # Version III
  zt' = (1 - at) (.) zt)  # multiplication by attention score
  #############################

  Ht = (1 - zt') (.) ht + zt' (.) Ht-1
```


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
  * **Default value**: *sigmoid* for f, *tanh* for g
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

* *direction*

  * **Description**: Specify if the RNN is forward, reverse, or bidirectional. If it is one of *forward* or *reverse* then `num_directions = 1`, if it is *bidirectional*, then `num_directions = 2`. This `num_directions` value specifies input/output shape requirements.
  * **Range of values**: *forward*, *reverse*, *bidirectional*
  * **Type**: `string`
  * **Required**: *yes*

* *linear_before_reset*

  * **Description**: *linear_before_reset* flag denotes, if the output of hidden gate is multiplied by the reset gate before or after linear transformation.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: `X` - 3D tensor of type *T1* `[batch_size, seq_length, input_size]`, input data. **Required.**

* **2**: `initial_hidden_state` - 3D tensor of type *T1* `[batch_size, num_directions, hidden_size]`, input hidden state data. **Required.**

* **3**: `sequence_lengths` - 1D tensor of type *T2* `[batch_size]`, specifies real sequence lengths for each batch element. **Required.**

* **4**: `W` - 3D tensor of type *T1* `[num_directions, 3 * hidden_size, input_size]`, the weights for matrix multiplication, gate order: zrh. **Required.**

* **5**: `R` - 3D tensor of type *T1* `[num_directions, 3 * hidden_size, hidden_size]`, the recurrence weights for matrix multiplication, gate order: zrh. **Required.**

* **6**: `B` - 2D tensor of type *T*. If *linear_before_reset* is set to 1, then the shape is `[num_directions, 4 * hidden_size]` - the sum of biases for z and r gates (weights and recurrence weights), the biases for h gate are placed separately. Otherwise the shape is `[num_directions, 3 * hidden_size]`, the sum of biases (weights and recurrence weights). **Required.**

* **7**: `A` - 3D tensor of type *T* `[batch_size, seq_length, 1]`, the attention score. **Required.**

**Outputs**

* **1**: `Y` - 4D tensor of type *T1* `[batch_size, num_directions, seq_length, hidden_size]`, concatenation of all the intermediate output values of the hidden.

* **2**: `Ho` - 3D tensor of type *T1* `[batch_size, num_directions, hidden_size]`, the last output value of hidden state.

**Types**

* *T1*: any supported floating-point type.
* *T2*: any supported integer type.

**Example**
```xml
<layer ... type="AUGRUSequence" ...>
    <data hidden_size="128"/>
    <input>
        <port id="0"> <!-- `X` input data -->
            <dim>1</dim>
            <dim>4</dim>
            <dim>16</dim>
        </port>
        <port id="1"> <!-- `initial_hidden_state` input -->
            <dim>1</dim>
            <dim>1</dim>
            <dim>128</dim>
        </port>
        <port id="2"> <!-- `sequence_lengths` input -->
            <dim>1</dim>
        </port>
         <port id="3"> <!-- `W` weights input -->
            <dim>1</dim>
            <dim>384</dim>
            <dim>16</dim>
        </port>
         <port id="4"> <!-- `R` recurrence weights input -->
            <dim>1</dim>
            <dim>384</dim>
            <dim>128</dim>
        </port>
         <port id="5"> <!-- `B` bias input -->
            <dim>1</dim>
            <dim>384</dim>
        </port>
        <port id="6"> <!-- `A` attention score input -->
            <dim>1</dim>
            <dim>4</dim>
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="7"> <!-- `Y` output -->
            <dim>1</dim>
            <dim>1</dim>
            <dim>4</dim>
            <dim>128</dim>
        </port>
        <port id="8"> <!-- `Ho` output -->
            <dim>1</dim>
            <dim>1</dim>
            <dim>128</dim>
        </port>
    </output>
</layer>
```
