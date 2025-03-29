Inverse
=======


.. meta::
  :description: Learn about Inverse-14 - a matrix operation that computes the inverse or adjoint for one matrix or a batch of input matrice.

**Versioned name**: *Inverse-14*

**Category**: *Matrix*

**Short description**: *Inverse* operation computes either the inverse or adjoint (conjugate transposes) of one or a batch of square invertible matrices.

**Detailed description**: *Inverse* operation computes the inverse of a square matrix.

The inverse matrix :math:`A^{-1}` of a square n-dimensional matrix :math:`A` is defined as:

.. math::

   A \cdot A^{-1} = A^{-1} \cdot A = I

where :math:`I` is the n-dimensional identity matrix.

The inverse matrix exists if and only if the input matrix is invertible (satisfies any of the properties of the *Invertible Matrix Theorem*). In that case, the inverse exists, and is unique. However, if the matrix is not invertible, the operation may raise an exception or return an undefined result. The operation may return slightly different results for the same input data on different devices due to parallelism and different data types implementations.

This operation can be used to compute the inverse of the adjoint matrix.

The adjoint matrix (conjugate transpose) :math:`adj(A)` of a square matrix of non-complex numbers :math:`A` is simply defined as:

.. math::

   adj(A) = A^{T}

where :math:`A^{T}` is the matrix transpose of :math:`A`.

With adjoint == true, the output of the algorithm can therefore be represented by (A^{T})^{-1}.

**Algorithm formulation**:

This operation uses LU decomposition with partial pivoting to compute the inverse (or adjoint) matrix.

.. note::

   LU decomposition decomposes matrix :math:`A` into 2 matrices :math:`L` and :math:`U`, where :math:`L` is the lower triangular matrix with all diagonal elements equal to 1 and :math:`U` is the upper triangular matrix.

.. math::

   A = L \cdot U

.. math::

   det(A) = det(L) * det(U) = 1 * det(U) = det(U)

.. note::

   To compute the inverse of :math:`A`, given its LU decomposition, it is enough to solve a set of n simple linear equations.
   A simple linear equation is of the form :math:`Ax=b`, where x and b are vectors, and x is the solution of the equation. The goal is to compute x, such that it is the i-th column of matrix :math:`A^{-1}`. To do so, let b be a vector of zeros, except for i-th spot that has a value of one (Then b is an i-th column of matrix I).
   It is easy to notice that the x-vectors create the matrix :math:`A^{-1}`, and the b-vectors create the Identity matrix.

.. math::

   A \cdot A^{-1} = I \implies A^{-1} = [x_1 \& x_2 \& ... \& x_n], A \cdot x_i = b_i, b_i \in col_i(I)

.. note::

   Using the LU decomposition, the simple linear equation can be replaced by two, even simpler linear equations.

.. math::

   A \cdot x = b \iff L \cdot U \cdot x = b \iff L \cdot y = b, U \cdot x = y

Algorithm pseudocode:

1. Start with original matrix :math:`A`. If the data type of :math:`A` is not f32, convert them to f32 to avoid accumulating rounding errors.
2. Copy initial matrix into matrix :math:`U`. Initialize matrix :math:`L` to be the Identity matrix (zero matrix with all diagonal elements set to 1).

   * If adjoint == true, set U to be the transpose of the initial matrix, instead of a direct copy.

3. Perform LU decomposition with partial pivoting.

   * Repeat this step for each column in the input matrix.
   * Let *c* be the index of the currently processed column.
   * Find the index of the row with the highest value in a given column - *pivot*.
   * If :math:`pivot \neq c`, swap the *pivot* and *c* row in :math:`L`. Repeat for :math:`U`.
   * Perform standard Gaussian elimination.

4. To obtain the inverse, solve for each column of :math:`A^{-1}` as explained above.

   * Solve linear equation :math:`Ly = b` for y (forward substitution)
   * Solve linear equation :math:`Ux = y` for x (backward substitution)
   * Set x as the corresponding column of the output inverse matrix :math:`A^{-1}`

5. Return the computed matrix. Convert it back from f32 to its original element type.

**Attribute**:

* *adjoint*

  * **Description**: Modifies the return value of the operation. If true, the operation returns the adjoint (conjugate transpose) of the input matrices instead of finding the inverse.
  * **Range of values**: `true`, `false`

    * ``true`` - output adjoint matrix.
    * ``false`` - output inverse matrix.

  * **Type**: `bool`
  * **Default value**: `false`
  * **Required**: *No*

**Input**:

* **1**: `input` - A tensor of shape [B1, B2, ..., Bn, ROW, COL] and type `T` representing the input square matrices. **The inner-most 2 dimensions form square matrices and must be of the same size.** B1, B2, ..., Bn represent any amount of batch dimensions (can be 0 for a single matrix input). **Required.**

**Output**:

* **1**: `output` - A tensor with the same type `T` as the input and same shape [B1, B2, ..., Bn, ROW, COL] as the input, representing the inverse matrices (or adjoint matrices) of the input matrices.

**Types**

* **T**: any supported floating-point type. Any type other than f32 will be converted to f32 before executing this op, and then converted back to the original input type to avoid accumulating rounding errors.

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
