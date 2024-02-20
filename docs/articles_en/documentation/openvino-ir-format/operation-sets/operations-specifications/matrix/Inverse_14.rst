.. {#openvino_docs_ops_matrix_Inverse_14}

Inverse
=======


.. meta::
  :description: Learn about Inverse-14 - a matrix operation that computes the inverse or adjoint for one matrix or a batch of input matrice.

**Versioned name**: *Inverse-14*

**Category**: *Matrix*

**Short description**: *Inverse* operation computes either the inverse or adjoint (conjugate transposes) of one or a batch of square invertible matrices.

**Detailed description**: *Inverse* operation computes the inverse of a square matrix. The operation uses LU decomposition with partial pivoting to compute the inverses.

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

**Algorithm formulation**:

.. note::

   LU decomposition decomposes matrix A into 2 matrices L and U, where L is the lower triangular matrix with all diagonal elements equal to 1 and U is the upper triangular matrix.

.. math::

   A = L \cdot U

.. note::

   LU decomposition allows to easily obtain determinant of A.

.. math::

   det(A) = det(L) * det(U) = 1 * det(U) (det(L) = 1, since L is a lower triangular matrix with diagonal elements set to 1)

.. note::

   To compute the inverse of A, given its LU decomposition, it is enough to solve multiple linear equations. 
   Set x to be a i-th column of matrix A^(-1). Set b to be a vector of zeros, except for i-th spot that has a value of one (in other words, b is a i-th column of matrix I).
   It is easy to notice that the set of x-columns creates the A^(-1) matrix, and the set of b-vectors creates the Identity matrix.

.. math::

   A \cdot A^{-1} = I
   <=> 
   A \cdot x = b, x - i-th column of A^{-1}, b - i-th column of I
   <=>
   L \cdot U \cdot x = b
   <=>
   L \cdot y = b, U \cdot x = y (y = U \cdot x)

Algorithm pseudocode:

1. Start with original matrix A.
2. Copy initial matrix into matrix U. Initialize matrix L to be the Identity matrix (zero matrix with all diagonal elements set to 1).
3. Perform LU decomposition with partial pivoting.

   * Repeat this step for each column in the input matrix.
   * Let *c* be the index of the currently proceseed column.
   * Find the index of the row with the highest value in a given column - *pivot*.
   * If *pivot* != *c*, swap the *pivot* and *c* row in L. Repeat for U. Note that this operations flips the sign of the determinant, so this has to be accounted for.
   * Perform standard Gaussian elimination.

4. To obtain the inverse, solve for each column of A^{-1} as explained above.

   * Solve linear equation Ly = b for y (forward substitution)
   * Solve linear equation Ux = y for x (backward substitution)
   * Set x as the corresponding column of the output inverse matrix A^(-1)

5. If adjoint == true, then it is necessary to multiply A^(-1) by its determinant.

   * As explained above, it is enough to compute det(U), since det(U) = det(A).
   * det(U) is just a multiplication of its diagonal elements.
   * Account for each row swap in the LU decomposition step - for every row swap, swap the sign of the dereminant.
   * Multiply all elements of A^(-1) by the determinant to obtain the adjugate matrix.

6. Return the computed matrix.

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
