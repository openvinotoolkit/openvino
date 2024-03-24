.. {#openvino_docs_ops_matrix_Inverse_14}

Inverse
=======


.. meta::
  :description: Learn about Inverse-14 - a matrix operation that computes the inverse or adjoint for one matrix or a batch of input matrice.

**Versioned name**: *Inverse-14*

**Category**: *Matrix*

**Short description**: *Inverse* operation computes either the inverse or adjoint (conjugate transposes) of one or a batch of square invertible matrices.

**Detailed description**: *Inverse* operation computes the inverse of a square matrix. The operation uses LU decomposition with partial pivoting to compute the inverses. (`LU decomposotion with partial pivoting algorithm <https://arxiv.org/abs/2304.03068>`__)

The inverse matrix A^(-1) of a square matrix A is defined as:

.. math::

   A \cdot A^{-1} = A^{-1} \cdot A = I

where **I** is the n-dimensional identity matrix.

The inverse matrix exists if and only if the input matrix is invertible. In that case, the inverse is unique. If the matrix is not invertible however, the operation may raise an exception or return an undefined result.

This operation can be used to compute the adjugate matrix instead of the inverse.

The adjugate matrix adj(A) of a square matrix A is defined as:

.. math::

   adj(A) = det(A) \cdot A^{-1}

where **A^{-1}** is the matrix inverse of A, and **det(A)** is the determinant of A.

The adjugate matrix exists if and only if the inverse matrix exists.

**Attribute**:

* *adjoint*

  * **Description**: Modifies the return value of the operation. If true, the operation returns the adjoint (conjugate transpose) of the input matrices instead of finding the inverse.
  * **Range of values**: `true`, `false` 

    * ``true`` - output adjugate matrix.
    * ``false`` - output inverse matrix. 

  * **Type**: `bool`
  * **Default value**: `false`
  * **Required**: *No*

**Input**:

* **1**: `input` - A tensor of shape [B1, B2, ..., Bn, ROW, COL] and type `T` representing the input square matrices. **The inner-most 2 dimensions form square matrices and must be of the same size.** B1, B2, ..., Bn represent any amount of batch dimensions (can be 0 for a single matrix input). **Required.**

**Output**:

* **1**: `output` - A tensor with the same type `T` as the input and same shape [B1, B2, ..., Bn, ROW, COL] as the input, representing the inverse matrices (or adjugate matrices) of the input matrices.

**Types**

* **T**: any supported floating-point type.

*Example 1: 2D input matrix.*

.. code-block:: xml
    :force:

    <layer ... name="Inverse" type="Inverse">
        <data/>
        <input>
            <port id="0" precision="FP32">
                <dim>3</dim> <!-- 3 rows of square matrix -->
                <dim>3</dim> <!-- 3 columns of square matrix -->
            </port>
        </input>
        <output>
            <port id="1" precision="FP32" names="Inverse:0">
                <dim>3</dim> <!-- 3 rows of square matrix -->
                <dim>3</dim> <!-- 3 columns of square matrix -->
            </port>
        </output>
    </layer>

*Example 2: 3D input tensor with one batch dimension and adjoint=true.*

.. code-block:: xml
    :force:

    <layer ... name="Inverse" type="Inverse">
        <data adjoint="true"/>
        <input>
            <port id="0" precision="FP32">
                <dim>2</dim> <!-- batch size of 2 -->
                <dim>4</dim> <!-- 4 rows of square matrix -->
                <dim>4</dim> <!-- 4 columns of square matrix -->
            </port>
        </input>
        <output>
            <port id="1" precision="FP32" names="Inverse:0">
                <dim>2</dim> <!-- batch size of 2 -->
                <dim>4</dim> <!-- 4 rows of square matrix -->
                <dim>4</dim> <!-- 4 columns of square matrix -->
            </port>
        </output>
    </layer>

*Example 3: 5D input tensor with three batch dimensions.*

.. code-block:: xml
    :force:

    <layer ... name="Inverse" type="Inverse">
        <data/>
        <input>
            <port id="0" precision="FP32">
                <dim>5</dim> <!-- batch size of 5 -->
                <dim>4</dim> <!-- batch size of 4 -->
                <dim>3</dim> <!-- batch size of 3 -->
                <dim>2</dim> <!-- 2 rows of square matrix -->
                <dim>2</dim> <!-- 2 columns of square matrix -->
            </port>
        </input>
        <output>
            <port id="1" precision="FP32" names="Inverse:0">
                <dim>5</dim> <!-- batch size of 5 -->
                <dim>4</dim> <!-- batch size of 4 -->
                <dim>3</dim> <!-- batch size of 3 -->
                <dim>2</dim> <!-- 2 rows of square matrix -->
                <dim>2</dim> <!-- 2 columns of square matrix -->
            </port>
        </output>
    </layer>
