RNNCell
=======


.. meta::
  :description: Learn about RNNCell-3 - a sequence processing operation, which
                can be performed on five required input tensors.

**Versioned name**: *RNNCell-3*

**Category**: *Sequence processing*

**Short description**: *RNNCell* represents a single RNN cell that computes the output using the formula described in the `article <https://hackernoon.com/understanding-architecture-of-lstm-cell-from-scratch-with-code-8da40f0b71f4>`__.

**Detailed description**:

*RNNCell* represents a single RNN cell and is part of  :doc:`RNNSequence <rnn-sequence-5>` operation.

.. code-block:: cpp

  Formula:
    *  - matrix multiplication
    ^T - matrix transpose
    f  - activation function
      Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)


**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *activations*

  * **Description**: activation functions for gates
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *tanh*
  * **Required**: *no*

* *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* functions attributes
  * **Range of values**: a list of floating-point numbers
  * **Type**: ``float[]``
  * **Default value**: None
  * **Required**: *no*

* *clip*

  * **Description**: *clip* specifies value for tensor clipping to be in *[-C, C]* before activations
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: *infinity* that means that the clipping is not applied
  * **Required**: *no*

**Inputs**

* **1**: ``X`` - 2D tensor of type *T* ``[batch_size, input_size]``, input data. **Required.**

* **2**: ``H`` - 2D tensor of type *T* ``[batch_size, hidden_size]``, initial hidden state. **Required.**

* **3**: ``W`` - 2D tensor of type *T* ``[hidden_size, input_size]``, the weights for matrix multiplication. **Required.**

* **4**: ``R`` - 2D tensor of type *T* ``[hidden_size, hidden_size]``, the recurrence weights for matrix multiplication. **Required.**

* **5**: ``B`` 1D tensor of type *T* ``[hidden_size]``, the sum of biases (weights and recurrence weights). **Required.**

**Outputs**

* **1**: ``Ho`` - 2D tensor of type *T* ``[batch_size, hidden_size]``, the last output value of hidden state.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="RNNCell" ...>
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
               <dim>128</dim>
               <dim>16</dim>
           </port>
           <port id="3">
               <dim>128</dim>
               <dim>128</dim>
           </port>
           <port id="4">
               <dim>128</dim>
           </port>
       </input>
       <output>
           <port id="5">
               <dim>1</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>

