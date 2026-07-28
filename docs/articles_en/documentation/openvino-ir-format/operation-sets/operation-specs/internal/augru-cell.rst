AUGRUCell
=========

**Versioned name**: *AUAUGRUCell*

**Category**: *Sequence processing*

**Short description**: *AUGRUCell* represents a single AUGRU Cell (GRU with attentional
update gate).

**Detailed description**: The main difference between *AUGRUCell* and
:doc:`GRUCell <../sequence/gru-cell-3>` is the additional attention score
input ``A``, which is a multiplier for the update gate.
The AUGRU formula is based on the `paper arXiv:1809.03672 <https://arxiv.org/abs/1809.03672>`__.

.. code-block:: py

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


**Attributes**

* *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *activations*

  * **Description**: activation functions for gates
  * **Range of values**: *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid* for f, *tanh* for g
  * **Required**: *no*

* *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* attributes of functions;
    applicability and meaning of these attributes depends on chosen activation functions
  * **Range of values**: []
  * **Type**: ``float[]``
  * **Default value**: []
  * **Required**: *no*

* *clip*

  * **Description**: *clip* specifies bound values *[-C, C]* for tensor clipping.
    Clipping is performed before activations.
  * **Range of values**: ``0.``
  * **Type**: ``float``
  * **Default value**: ``0.`` that means the clipping is not applied
  * **Required**: *no*

* *linear_before_reset*

  * **Description**: *linear_before_reset* flag denotes, if the output of hidden gate
    is multiplied by the reset gate before or after linear transformation.
  * **Range of values**: False
  * **Type**: ``boolean``
  * **Default value**: False
  * **Required**: *no*.

**Inputs**

* **1**: ``X`` - 2D tensor of type *T* and shape ``[batch_size, input_size]``, input
  data. **Required.**

* **2**: ``H_t`` - 2D tensor of type *T* and shape ``[batch_size, hidden_size]``.
  Input with initial hidden state data. **Required.**

* **3**: ``W`` - 2D tensor of type *T* and shape ``[3 * hidden_size, input_size]``.
  The weights for matrix multiplication, gate order: zrh. **Required.**

* **4**: ``R`` - 2D tensor of type *T* and shape ``[3 * hidden_size, hidden_size]``.
  The recurrence weights for matrix multiplication, gate order: zrh. **Required.**

* **5**: ``B`` - 2D tensor of type *T*. The biases. If *linear_before_reset* is set
  to ``False``, then the shape is ``[3 * hidden_size]``, gate order: zrh. Otherwise
  the shape is ``[4 * hidden_size]`` - the sum of biases for z and r gates (weights and
  recurrence weights), the biases for h gate are placed separately. **Required.**

* **6**: ``A`` - 2D tensor of type *T* and shape ``[batch_size, 1]``, the attention
  score. **Required.**


**Outputs**

* **1**: ``Ho`` - 2D tensor of type *T* ``[batch_size, hidden_size]``, the last output
  value of hidden state.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="AUGRUCell" ...>
       <data hidden_size="128"/>
        <input>
           <port id="0"> <!-- `X` input data -->
               <dim>1</dim>
               <dim>16</dim>
           </port>
           <port id="1"> <!-- `H_t` input -->
               <dim>1</dim>
               <dim>128</dim>
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

