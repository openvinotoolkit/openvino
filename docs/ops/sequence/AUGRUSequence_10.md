# AUGRUSequence  {#openvino_docs_ops_sequence_AUGRUSequence_10}

**Versioned name**: *AUGRUSequence-10*

**Category**: *Sequence processing*

**Short description**: *AUGRUSequence* operation represents a series of AUGRU cells (GRU with attentional update gate).

**Detailed description**: The main difference between *AUGRUSequence* and [GRUSequence](./GRUSequence_5.md) is the additional attention score input `A`, which is a multiplier for the update gate.
The AUGRU formula is based on the [paper arXiv:1809.03672](https://arxiv.org/abs/1809.03672).

```
AUGRU formula:
  *  - matrix multiplication
 (.) - Hadamard product (element-wise)

 f, g - activation functions
 z - update gate, r - reset gate, h - hidden gate
 a - attention score

  rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)  # 'linear_before_reset' is False

  zt' = (1 - at) (.) zt  # multiplication by attention score

  Ht = (1 - zt') (.) ht + zt' (.) Ht-1
```

Activation functions for gates: *sigmoid* for f, *tanh* for g.
Only `forward` direction is supported.

**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Required**: *yes*


**Inputs**

* **1**: `X` - 3D tensor of type *T1* `[batch_size, seq_length, input_size]`, input data. **Required.**

* **2**: `initial_hidden_state` - 2D tensor of type *T1* and shape `[batch_size, hidden_size]`, input hidden state data. **Required.**

* **3**: `sequence_lengths` - 1D tensor of type *T2* and shape `[batch_size]`, specifies real sequence lengths for each batch element. **Required.**

* **4**: `W` - 2D tensor of type *T1* and shape `[3 * hidden_size, input_size]`, the weights for matrix multiplication, gate order: zrh. **Required.**

* **5**: `R` - 2D tensor of type *T1* and shape `[3 * hidden_size, hidden_size]`, the recurrence weights for matrix multiplication, gate order: zrh. **Required.**

* **6**: `B` - 1D tensor of type *T1* and shape `[3 * hidden_size]`, the biases, gate order: zrh. **Required.**

* **7**: `A` - 3D tensor of type *T1* `[batch_size, seq_length, 1]`, the attention score. **Required.**

**Outputs**

* **1**: `Y` - 4D tensor of type *T1* `[batch_size, seq_length, hidden_size]`, concatenation of all the intermediate output values of the hidden.

* **2**: `Ho` - 3D tensor of type *T1* `[batch_size, hidden_size]`, the last output value of hidden state.

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
            <dim>128</dim>
        </port>
        <port id="2"> <!-- `sequence_lengths` input -->
            <dim>1</dim>
        </port>
         <port id="3"> <!-- `W` weights input -->
            <dim>384</dim>
            <dim>16</dim>
        </port>
         <port id="4"> <!-- `R` recurrence weights input -->
            <dim>384</dim>
            <dim>128</dim>
        </port>
         <port id="5"> <!-- `B` bias input -->
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
            <dim>4</dim>
            <dim>128</dim>
        </port>
        <port id="8"> <!-- `Ho` output -->
            <dim>1</dim>
            <dim>128</dim>
        </port>
    </output>
</layer>
```
