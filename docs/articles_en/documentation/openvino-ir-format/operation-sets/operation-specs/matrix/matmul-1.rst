MatMul
======


.. meta::
  :description: Learn about MatMul-1 - a matrix multiplication operation,
                which can be performed on two required input tensors.

**Versioned name**: *MatMul-1*

**Category**: *Matrix multiplication*

**Short description**: Generalized matrix multiplication

**Detailed description**

*MatMul* operation takes two tensors and performs usual matrix-matrix multiplication, matrix-vector multiplication or vector-matrix multiplication depending on argument shapes. Input tensors can have any rank >= 1. Two right-most axes in each tensor are interpreted as matrix rows and columns dimensions while all left-most axes (if present) are interpreted as multi-dimensional batch: ``[BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K, ROW_INDEX_DIM, COL_INDEX_DIM]``. The operation supports usual broadcast semantics for batch dimensions. It enables multiplication of batch of pairs of matrices in a single shot.

Before matrix multiplication, there is an implicit shape alignment for input arguments. It consists of the following steps:

1. Applying transpositions specified by optional ``transpose_a`` and ``transpose_b`` attributes. Only the two right-most dimensions are transposed, other dimensions remain the same. Transpose attributes are ignored for 1D tensors.

2. One-dimensional tensors unsqueezing is applied for each input independently. The axes inserted in this step are not included in the output shape.

   * If rank of the **first** input is equal to 1, it is always unsqueezed to 2D tensor **row vector** (regardless of ``transpose_a``) by adding axes with size 1 at ROW_INDEX_DIM, to the **left** of the shape. For example ``[S]`` will be reshaped to ``[1, S]``.
   * If rank of the **second** input is equal to 1, it is always unsqueezed to 2D tensor **column vector** (regardless of ``transpose_b``) by adding axes with size 1 at COL_INDEX_DIM, to the **right** of the shape. For example ``[S]`` will be reshaped to ``[S, 1]``.

3. If ranks of input arguments are different after steps 1 and 2, the tensor with a smaller rank is unsqueezed from the left side of the shape by necessary number of axes to make both shapes of the same rank.

4. Usual rules of the broadcasting are applied for batch dimensions.

Temporary axes inserted in **step 2** are removed from the final output shape after multiplying.
After vector-matrix multiplication, the temporary axis inserted at ROW_INDEX_DIM is removed. After matrix-vector multiplication, the temporary axis inserted at COL_INDEX_DIM is removed.
Output shape of two 1D tensors multiplication ``[S] x [S]`` is squeezed to scalar.

Output shape inference logic examples (ND here means bigger than 1D):

* 1D x 1D: ``[X] x [X] -> [1, X] x [X, 1] -> [1, 1] => []`` (scalar)
* 1D x ND: ``[X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]``
* ND x 1D: ``[B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] => [B, ..., X]``
* ND x ND: ``[B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]``


Two attributes, ``transpose_a`` and ``transpose_b`` specify embedded transposition for two right-most dimensions for the first and the second input tensors correspondingly. It implies swapping of ROW_INDEX_DIM and COL_INDEX_DIM in the corresponding input tensor. Batch dimensions and 1D tensors are not affected by these attributes.

**Attributes**

* *transpose_a*

  * **Description**: transposes dimensions ROW_INDEX_DIM and COL_INDEX_DIM of the 1st input; **false** means no transpose, **true** means transpose. It is ignored if first input is 1D tensor.
  * **Range of values**: false or true
  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*

* *transpose_b*

  * **Description**: transposes dimensions ROW_INDEX_DIM and COL_INDEX_DIM of the 2nd input; **false** means no transpose, **true** means transpose. It is ignored if second input is 1D tensor.
  * **Range of values**: false or true
  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*


**Inputs**:

* **1**: Tensor of type *T* with matrices A. Rank >= 1. **Required.**

* **2**: Tensor of type *T* with matrices B. Rank >= 1. **Required.**

**Outputs**

* **1**: Tensor of type *T* with results of the multiplication.

**Types**:

* *T*: any supported floating-point or integer type.

**Example**

*Vector-matrix multiplication*

.. code-block:: xml
   :force:

   <layer ... type="MatMul">
       <input>
           <port id="0">
               <dim>1024</dim>
           </port>
           <port id="1">
               <dim>1024</dim>
               <dim>1000</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>1000</dim>
           </port>
       </output>
   </layer>


*Matrix-vector multiplication*

.. code-block:: xml
   :force:

   <layer ... type="MatMul">
       <input>
           <port id="0">
               <dim>1000</dim>
               <dim>1024</dim>
           </port>
           <port id="1">
               <dim>1024</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>1000</dim>
           </port>
       </output>
   </layer>


*Matrix-matrix multiplication (like FullyConnected with batch size 1)*

.. code-block:: xml
   :force:

   <layer ... type="MatMul">
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>1024</dim>
           </port>
           <port id="1">
               <dim>1024</dim>
               <dim>1000</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>1</dim>
               <dim>1000</dim>
           </port>
       </output>
   </layer>


*Vector-matrix multiplication with embedded transposition of the second matrix*

.. code-block:: xml
   :force:

   <layer ... type="MatMul">
       <data transpose_b="true"/>
       <input>
           <port id="0">
               <dim>1024</dim>
           </port>
           <port id="1">
               <dim>1000</dim>
               <dim>1024</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>1000</dim>
           </port>
       </output>
   </layer>


*Matrix-matrix multiplication (like FullyConnected with batch size 10)*

.. code-block:: xml
   :force:

   <layer ... type="MatMul">
       <input>
           <port id="0">
               <dim>10</dim>
               <dim>1024</dim>
           </port>
           <port id="1">
               <dim>1024</dim>
               <dim>1000</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>10</dim>
               <dim>1000</dim>
           </port>
       </output>
   </layer>


*Multiplication of batch of 5 matrices by a one matrix with broadcasting*

.. code-block:: xml
   :force:

   <layer ... type="MatMul">
       <input>
           <port id="0">
               <dim>5</dim>
               <dim>10</dim>
               <dim>1024</dim>
           </port>
           <port id="1">
               <dim>1024</dim>
               <dim>1000</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>5</dim>
               <dim>10</dim>
               <dim>1000</dim>
           </port>
       </output>
   </layer>



