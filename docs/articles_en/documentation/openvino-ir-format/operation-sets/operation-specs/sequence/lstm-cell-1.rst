LSTMCell
========


.. meta::
  :description: Learn about LSTMCell-1 - a sequence processing operation, which
                can be performed on five required and one optional input tensor.

**Versioned name**: *LSTMCell-1*

**Category**: *Sequence processing*

**Short description**: *LSTMCell* operation represents a single LSTM cell. It computes the output using the formula described in the original paper `Long Short-Term Memory <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf>`__.

**Detailed description**: *LSTMCell* computes the output *Ht* and *ot* for current time step based on the following formula:

.. code-block:: sh

   Formula:
     *  - matrix multiplication
    (.) - Hadamard product (element-wise)
    [,] - concatenation
    f, g, h - are activation functions.
        it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
        ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        Ct = ft (.) Ct-1 + it (.) ct
        ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
        Ht = ot (.) h(Ct)


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
  * **Default value**: *sigmoid* for f, *tanh* for g, *tanh* for h
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

**Inputs**

* **1**: ``X`` - 2D tensor of type *T* ``[batch_size, input_size]``, input data. **Required.**

* **2**: ``initial_hidden_state`` - 2D tensor of type *T* ``[batch_size, hidden_size]``. **Required.**

* **3**: ``initial_cell_state`` - 2D tensor of type *T* ``[batch_size, hidden_size]``. **Required.**

* **4**: ``W`` - 2D tensor of type *T* ``[4 * hidden_size, input_size]``, the weights for matrix multiplication, gate order: fico. **Required.**

* **5**: ``R`` - 2D tensor of type *T* ``[4 * hidden_size, hidden_size]``, the recurrence weights for matrix multiplication, gate order: fico. **Required.**

* **6**: ``B`` 1D tensor of type *T* ``[4 * hidden_size]``, the sum of biases (weights and recurrence weights), if not specified - assumed to be 0. **optional.**


**Outputs**

* **1**: ``Ho`` - 2D tensor of type *T* ``[batch_size, hidden_size]``, the last output value of hidden state.

* **2**: ``Co`` - 2D tensor of type *T* ``[batch_size, hidden_size]``, the last output value of cell state.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

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



