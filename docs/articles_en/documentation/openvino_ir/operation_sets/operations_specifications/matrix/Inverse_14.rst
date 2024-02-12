.. {#openvino_docs_ops_matrix_Inverse_14}

Inverse
=======


.. meta::
  :description: Learn about Inverse-14 - a matrix operation that computes the inverse or adjoint of one or multiple input matrices.

**Versioned name**: *Inverse-14*

**Category**: *Linear Algebra*

**Short description**: *Inverse* operation computes either the inverse or adjoint (conjugate transposes) of one or more square invertible matrices.

**Detailed description**: *Inverse* operation computes the inverse of a square matrix. The operation uses LU decomposition with partial pivoting to compute the inverses.

The inverse matrix A^(-1) of a square matrix A is defined as:

\[ A \cdot A^{-1} = A^{-1} \cdot A = I \]

where \( I \) is the n-dimensional identity matrix.

The inverse matrix exists if and only if the input matrix is invertible. In that case, the inverse is unique. If the matrix is not invertible however, the operation may raise an exception or return an undefined result.

This operation can be used to compute the adjugate matrix instead of the inverse.

The adjugate matrix adj(A) of a square matrix A is defined as:

\[ adj(A) = det(A) \cdot A^{-1} \]

where \( A^{-1} \) is the matrix inverse of A, and \( det(A) \) is the determinant of A.

The adjugate matrix exists if and only if the inverse matrix exists.

**Inputs**:

* **1**: `input` - A tensor of shape [..., M, M] and type `T_IN` representing the input square matrices. The inner-most 2 dimensions form square matrices and must be of the same size. **Required.**

**Outputs**:

* **1**: `output` - A tensor with the same type `T_IN` as the input and same shape [..., M, M] as the input, representing the inverse matrices (or adjugate matrices) of the input matrices.

**Attributes**:

*  ``adjoint``

    * **Description**: Modifies the return value of the operation. If true, the operation returns the adjoint (conjugate transpose) of the input matrices instead of finding the inverse.
    * **Range of values**: `true`, `false`

        * ``true`` - output adjugate matrix.
        * ``false`` - output inverse matrix.

    * **Type**: `bool`
    * **Default value**: `false`
    * **Required**: *No*

**Types**

* **T_IN**: any supported floating-point type.

**Example**:

.. code-block:: xml
    :force:

    <layer ... name="Inverse" type="Inverse">
        <data />
        <input>
            <port id="0" precision="FP32">
                <dim>2</dim> <!-- batch size of 2 -->
                <dim>3</dim> <!-- matrix size of 3x3 -->
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="1" precision="FP32" names="Inverse:0">
                <dim>2</dim> <!-- batch size of 2 -->
                <dim>3</dim> <!-- matrix size of 3x3 -->
                <dim>3</dim>
            </port>
        </output>
    </layer>
