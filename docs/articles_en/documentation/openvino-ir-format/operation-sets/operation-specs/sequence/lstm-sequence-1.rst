LSTMSequence
============


.. meta::
  :description: Learn about LSTMSequence-1 - a sequence processing operation, which
                can be performed on seven required input tensors.

**Versioned name**: *LSTMSequence-1*

**Category**: *Sequence processing*

**Short description**: *LSTMSequence* operation represents a series of LSTM cells. Each cell is implemented as :doc:`LSTM Cell <lstm-cell-1>` operation.

**Detailed description**

A single cell in the sequence is implemented in the same way as in :doc:`LSTM Cell <lstm-cell-1>` operation. *LSTMSequence* represents a sequence of LSTM cells. The sequence can be connected differently depending on ``direction`` attribute that specifies the direction of traversing of input data along sequence dimension or specifies whether it should be a bidirectional sequence. The most of the attributes are in sync with the specification of ONNX LSTM operator defined `LSTMCell <https://github.com/onnx/onnx/blob/master/docs/Operators.md#lstm>`__ .


**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: ``int``
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
  * **Type**: ``float[]``
  * **Default value**: None
  * **Required**: *no*

* *clip*

  * **Description**: *clip* specifies bound values *[-C, C]* for tensor clipping. Clipping is performed before activations.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: *infinity* that means that the clipping is not applied
  * **Required**: *no*

* *direction*

  * **Description**: Specify if the RNN is forward, reverse, or bidirectional. If it is one of *forward* or *reverse* then ``num_directions = 1``, if it is *bidirectional*, then ``num_directions = 2``. This ``num_directions`` value specifies input/output shape requirements.
  * **Range of values**: *forward*, *reverse*, *bidirectional*
  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**

* **1**: ``X`` - 3D tensor of type *T1* ``[batch_size, seq_length, input_size]``, input data. It differs from LSTMCell 1st input only by additional axis with size ``seq_length``. **Required.**

* **2**: ``initial_hidden_state`` - 3D tensor of type *T1* ``[batch_size, num_directions, hidden_size]``, input hidden state data. **Required.**

* **3**: ``initial_cell_state`` - 3D tensor of type *T1* ``[batch_size, num_directions, hidden_size]``, input cell state data. **Required.**

* **4**: ``sequence_lengths`` - 1D tensor of type *T2* ``[batch_size]``, specifies real sequence lengths for each batch element. In case of negative values in this input, the operation behavior is undefined. **Required.**

* **5**: ``W`` - 3D tensor of type *T1* ``[num_directions, 4 * hidden_size, input_size]``, the weights for matrix multiplication, gate order: fico. **Required.**

* **6**: ``R`` - 3D tensor of type *T1* ``[num_directions, 4 * hidden_size, hidden_size]``, the recurrence weights for matrix multiplication, gate order: fico. **Required.**

* **7**: ``B`` - 2D tensor of type *T1* ``[num_directions, 4 * hidden_size]``, the sum of biases (weights and recurrence weights). **Required.**

**Outputs**

* **1**: ``Y`` – 3D tensor of type *T1* ``[batch_size, num_directions, seq_len, hidden_size]``, concatenation of all the intermediate output values of the hidden.

* **2**: ``Ho`` - 3D tensor of type *T1* ``[batch_size, num_directions, hidden_size]``, the last output value of hidden state.

* **3**: ``Co`` - 3D tensor of type *T1* ``[batch_size, num_directions, hidden_size]``, the last output value of cell state.

**Types**

* *T1*: any supported floating-point type.
* *T2*: any supported integer type.

**Example**

.. code-block:: cpp

   <layer ... type="LSTMSequence" ...>
       <data hidden_size="128"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>4</dim>
               <dim>16</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>1</dim>
               <dim>128</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>1</dim>
               <dim>128</dim>
           </port>
           <port id="3">
               <dim>1</dim>
           </port>
            <port id="4">
               <dim>1</dim>
               <dim>512</dim>
               <dim>16</dim>
           </port>
            <port id="5">
               <dim>1</dim>
               <dim>512</dim>
               <dim>128</dim>
           </port>
            <port id="6">
               <dim>1</dim>
               <dim>512</dim>
           </port>
       </input>
       <output>
           <port id="7">
               <dim>1</dim>
               <dim>1</dim>
               <dim>4</dim>
               <dim>128</dim>
           </port>
           <port id="8">
               <dim>1</dim>
               <dim>1</dim>
               <dim>128</dim>
           </port>
           <port id="9">
               <dim>1</dim>
               <dim>1</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>

