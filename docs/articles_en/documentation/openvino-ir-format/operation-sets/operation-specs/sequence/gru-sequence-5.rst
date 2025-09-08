GRUSequence
===========


.. meta::
  :description: Learn about GRUSequence-5 - a sequence processing operation, which
                can be performed on six required input tensors.

**Versioned name**: *GRUSequence-5*

**Category**: *Sequence processing*

**Short description**: *GRUSequence* operation represents a series of GRU cells. Each cell is implemented as GRUCell operation.

**Detailed description**

A single cell in the sequence is implemented in the same way as in *GRUCell* operation. *GRUSequence*
represents a sequence of GRU cells. The sequence can be connected differently depending on
``direction`` attribute that specifies the direction of traversing of input data along sequence
dimension or specifies whether it should be a bidirectional sequence. The most of the attributes
are in sync with the specification of ONNX GRU operator defined
`GRUCell <https://github.com/onnx/onnx/blob/main/docs/Operators.md#gru>`__


**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Required**: *yes*

* *activations*

  * **Description**: *activations* specifies activation functions for gates, there are two gates,
    so two activation functions should be specified as a value for this attributes
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid,tanh*
  * **Required**: *no*

* *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* attributes of functions;
    applicability and meaning of these attributes depends on chosen activation functions
  * **Range of values**: a list of floating-point numbers
  * **Type**: ``float[]``
  * **Default value**: None
  * **Required**: *no*

* *clip*

  * **Description**: *clip* specifies bound values *[-C, C]* for tensor clipping. Clipping is performed before activations.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: *infinity* that means that the clipping is not applied
  * **Required**: *no*

* *direction*

  * **Description**: Specify if the RNN is forward, reverse, or bidirectional. If it is one of *forward* or *reverse*
    then ``num_directions = 1``, if it is *bidirectional*, then ``num_directions = 2``. This ``num_directions``
    value specifies input/output shape requirements.
  * **Range of values**: *forward*, *reverse*, *bidirectional*
  * **Type**: ``string``
  * **Required**: *yes*

* *linear_before_reset*

  * **Description**: *linear_before_reset* flag denotes if the layer behaves according to the modification
    of *GRUCell* described in the formula in the `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU>`__.
  * **Range of values**: True or False
  * **Type**: ``boolean``
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: ``X`` - 3D tensor of type *T1* ``[batch_size, seq_length, input_size]``, input data.
  It differs from GRUCell 1st input only by additional axis with size ``seq_length``. **Required.**
* **2**: ``initial_hidden_state`` - 3D tensor of type *T1* ``[batch_size, num_directions, hidden_size]``,
  input hidden state data. **Required.**
* **3**: ``sequence_lengths`` - 1D tensor of type *T2* ``[batch_size]``, specifies real sequence lengths
  for each batch element. In case of negative values in this input, the operation behavior is undefined. **Required.**
* **4**: ``W`` - 3D tensor of type *T1* ``[num_directions, 3 * hidden_size, input_size]``,
  the weights for matrix multiplication, gate order: zrh. **Required.**
* **5**: ``R`` - 3D tensor of type *T1* ``[num_directions, 3 * hidden_size, hidden_size]``,
  the recurrence weights for matrix multiplication, gate order: zrh. **Required.**
* **6**: ``B`` - 2D tensor of type *T*. If *linear_before_reset* is set to 1, then the shape
  is ``[num_directions, 4 * hidden_size]`` - the sum of biases for z and r gates (weights and recurrence weights),
  the biases for h gate are placed separately. Otherwise the shape is ``[num_directions, 3 * hidden_size]``,
  the sum of biases (weights and recurrence weights). **Required.**

**Outputs**

* **1**: ``Y`` - 4D tensor of type *T1* ``[batch_size, num_directions, seq_len, hidden_size]``, concatenation of all the intermediate output values of the hidden.
* **2**: ``Ho`` - 3D tensor of type *T1* ``[batch_size, num_directions, hidden_size]``, the last output value of hidden state.

**Types**

* *T1*: any supported floating-point type.
* *T2*: any supported integer type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="GRUSequence" ...>
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
           </port>
            <port id="3">
               <dim>1</dim>
               <dim>384</dim>
               <dim>16</dim>
           </port>
            <port id="4">
               <dim>1</dim>
               <dim>384</dim>
               <dim>128</dim>
           </port>
            <port id="5">
               <dim>1</dim>
               <dim>384</dim>
           </port>
       </input>
       <output>
           <port id="6">
               <dim>1</dim>
               <dim>1</dim>
               <dim>4</dim>
               <dim>128</dim>
           </port>
           <port id="7">
               <dim>1</dim>
               <dim>1</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>



