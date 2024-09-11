Einsum
======


.. meta::
  :description: Learn about Einsum-7 - a matrix multiplication operation,
                which can be performed on multiple input tensors of different shape.

**Versioned name**: *Einsum-7*

**Category**: *Matrix multiplication*

**Short description**: *Einsum* performs the Einstein summation convention on the operands.

**Detailed description**: *Einsum* can represent many common multidimensional linear algebraic tensor operations: matrix multiplication;
inner (or dot), outer and cross products; transpose; trace and diagonal extraction.
Also, a single *Einsum* operation can express complex combination of these common linear algebraic tensor operations on multiple operands,
for example, a dot product of a diagonal, extracted from a tensor with shape ``[5, 5]``, and 5D vector is performed by single Einsum operation.
The Einstein summation convention on input tensors is defined by ``equation``, which is a mandatory attribute of *Einsum* operation.
The operation supports ``equation`` in explicit and implicit modes. The formats of ``equation`` in both modes are described below.

In explicit mode, the einsum ``equation`` has the output subscript separated from the input subscripts by ``->``, and has the following format for ``n`` operands:
``<subscript for input1>, <subscript for input2>, ..., <subscript for inputn> -> <subscript for output>``.
Each input subscript ``<subscript for input1>`` contains a sequence of labels (alphabetic letters ``['A',...,'Z','a',...,'z']``),
where each label refers to a dimension of the corresponding operand. Labels are case sensitive and capital letters precede lowercase letters in alphabetical sort.
Labels do not need to appear in a subscript in alphabetical order.
The subscript for a scalar input is empty. The input subscripts are separated with a comma ``,``.
The output subscript ``<subscript for output>`` represents a sequence of labels (alphabetic letters ``['A',...,'Z','a',...,'z']``).
The length of an input subscript matches a rank of the input. The input subscript is empty for a scalar input.

*Einsum* operation on multiple inputs can be treated as several consecutive *Einsum* operations. In the first step, *Einsum* applies the first two inputs.
In the second step, it operates on the result of the first step and the third input, and so forth.
*Einsum* operates on two operands similar to element-wise multiplication by all pairs of batches from both operands.
The batch dimensions are defined with labels belonging to only one of the two input subscripts.

For example, the intermediate result after the first step for *Einsum* with three inputs of shapes ``[2, 5]``, ``[5, 3, 6]`` and ``[5, 3]``,
and ``equation`` equal to ``ab,bcd,bc->ca`` will be a tensor of shape ``[2, 5, 3, 6]`` with a subscript ``abcd``,
where batch dimensions for the first input and the second input are represented with label sequences ``a`` and ``cd``.
The next step performs the same logic on input tensors of shapes ``[2, 5, 3, 6]`` and ``[5, 3]`` with subscripts ``abcd`` and ``bc``, and
outputs a tensor of shape ``[2, 5, 3, 6]`` with a subscript ``abcd``.
Lastly, the output subscript defines the order of output dimensions, and sum-reduced dimensions.
Dimensions corresponding to absent labels in the output subscript are sum-reduced. The final result for the considered example is of shape equal to ``[3,2]``,
where dimensions with labels ``b`` and ``d`` are reduced, and the transpose is applied to get output layout ``ca``.

.. note::

   * *Einsum* operation can perform on a single operand. In this case, the operation can transpose the input and reduce its dimensions.
   * Input ranks must be equal to the length of corresponding subscripts. Dimensions with the same corresponding labels in input subscripts must be equal in size.
   * A label can be repeated in the same input subscript, for example, ``equation`` equal to ``aac,abd,ddde``. In this case, the corresponding dimensions must match in size, and the operand is replaced by its diagonal along these dimensions. For example, *Einsum* operation on the single 3D tensor of shape ``[2, 4, 5, 4]`` with ``equation`` equal to ``ijkj->ij``.
   * The specification considers the primitive algorithm for *Einsum* operation for better understanding of the operation and does not recommend it for implementation.
   * The described algorithm can be improved by immediate dimension sum-reduction of the intermediate results if the corresponding labels are absent  in the input subscripts of subsequent inputs and the output subscript. It can significantly boost performance and reduce memory costs. In the considered example, after the first step you can reduce the dimension corresponding to the label ``d``.

The output shape is computed by concatenation of dimension sizes to which labels in the output subscript correspond in the specified order.

Example 1 shows how *Einsum* computes inner product of two 1D tensors:

.. code-block:: cpp

   a1 = [1.0, 2.0, 3.0]
   a2 = [4.0, 5.0, 6.0]
   equation = "i,i->"
   output = 32.0

Example 2 shows how *Einsum* computes matrix-vector multiplication:

.. code-block:: cpp

   A = [[1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]]
   b = [4.0, 5.0, 6.0]
   equation = "ij,j->i"
   output = [32.0, 32.0]

Example 3 shows how *Einsum* computes a trace for each batch object:

.. code-block:: cpp

   A = [[[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]],
        [[2.0, 4.0, 6.0],
         [8.0, 10.0, 12.0],
         [14.0, 16.0, 18.0]]]
   equation = "kii->k"
   output = [15.0, 30.0]

Example 4 shows how *Einsum* extracts a diagonal for each batch object:

.. code-block:: cpp

   A = [[[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]],
        [[2.0, 4.0, 6.0],
         [8.0, 10.0, 12.0],
         [14.0, 16.0, 18.0]]]
   equation = "kii->ki"
   output = [[1.0, 5.0, 9.0],
             [2.0, 10.0, 18.0]]

Example 5 shows how *Einsum* transposes input tensor:

.. code-block:: cpp

   A = [[[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]]]
   equation = "ijk->kij"
   output = [[[1.0, 4.0, 7.0]],
             [[2.0, 5.0, 8.0]],
             [[3.0, 6.0, 9.0]]]


In addition to an alphabetic label, ellipsis ``...`` can be used as a label in a subscript to cover broadcasted dimensions. Each input subscript can contain at most one ellipsis. For example, the ellipsis in input subscript ``a...bc`` for five rank tensor covers the second and third dimensions. In case input subscripts contain ellipsis for several operands, the dimensions covered by the ellipsis must be broadcastable to satisfy numpy broadcasting (or multidirectional broadcasting) rules available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`. If at least one input subscript contains an ellipsis, the output subscript must always contain one ellipsis. For example, *Einsum* operation on two inputs of shapes ``[9, 1, 4, 3]`` and ``[3, 11, 7, 1]`` with ``equation="a...b,b...->a..."`` has ellipsis for both operands covering dimensions with sizes ``[1, 4]`` and ``[11, 7, 1]`` that are broadcasted to ``[11, 7, 4]``. The resulted shape of *Einsum* operation will be ``[9, 11, 7, 4]`` since the dimension labeled with ``a`` is left with broadcasted dimensions.

Example 6 shows how *Einsum* operates on the single input with an equation containing ellipsis:

.. code-block:: cpp

   A = [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]]
   equation = "a...->..."
   output = [12.0, 15.0, 18.0]

Example 7 shows how *Einsum* operates with broadcasting two operands:

.. code-block:: cpp

   A = [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]]
   B = [0.5]
   equation = "a...,...->a..."
   output = [[0.5, 1.0, 1.5],
             [2.0, 2.5, 3.0],
             [3.5, 4.0, 4.5]]

In implicit mode (a classical form of Einstein summation), the equation does not have the output subscript and has the following format:
``<subscript for input1>, <subscript for input2>, ..., <subscript for inputn>``.
The equation in implicit mode consists of only input subscripts for each operand.
The output subscript can be recovered as a sequence of alphabetically sorted labels that are not repeated in the left-hand side of the equation.
For example, ``equation = "dbbc,ca"`` in implicit mode is equivalent to ``equation = "dbbc,ca->ad"`` in explicit mode.
The equation in implicit mode can set up only subset of Einstein summation conventions. For example, ``equation = "kii->i"`` cannot be represented in implicit mode.
In case ellipsis label is in the left-hand side of the equation in implicit mode, the ellipsis comes first in the output subscript for the recovery.

Example 8 shows how *Einsum* operates with an equation containing both capital and lowercase letters in implicit mode
``equation = "AbC"`` that is the same as ``equation = "AbC->ACb"``:

.. code-block:: cpp

   A = [[[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]]
   equation = "AbC"
   output = [[[1.0, 4.0],
              [2.0, 5.0],
              [3.0, 6.0]]]

.. note::

   The equation in both modes can contain blank space characters (U+0020) at any positions that can be removed without losing equivalence.

**Attributes**:

* *equation*

  * **Description**: it defines Einstein summation convention on input operands. The equation must be in either explicit or implicit mode.
  * **Range of values**: the equation format is described above
  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**:

* **Multiple inputs**: Tensors of type *T* and different shapes.

**Output**:

* **1**: Tensor of type *T* and shape is computed based on the output subscript of the equation.

**Types**

* *T*: any numeric type.

**Examples**

.. code-block:: xml
   :force:

   <layer ... type="Einsum" version="opset7">
       <data equation="ij,ij->i"/>
       <input>
           <port id="0">
               <dim>2</dim>
               <dim>64</dim>
           </port>
           <port id="0">
               <dim>2</dim>
               <dim>64</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>2</dim>
           </port>
       </output>
   </layer>

.. code-block:: xml
   :force:

   <layer ... type="Einsum" version="opset7">
       <data equation="ab...,ac...,ade->...bc"/>
       <input>
           <port id="0">
               <dim>2</dim>
               <dim>3</dim>
               <dim>4</dim>
           </port>
           <port id="1">
               <dim>2</dim>
               <dim>7</dim>
               <dim>1</dim>
           </port>
           <port id="3">
               <dim>2</dim>
               <dim>4</dim>
               <dim>7</dim>
           </port>
       </input>
       <output>
           <port id="4">
               <dim>4</dim>
               <dim>3</dim>
               <dim>7</dim>
           </port>
       </output>
   </layer>


