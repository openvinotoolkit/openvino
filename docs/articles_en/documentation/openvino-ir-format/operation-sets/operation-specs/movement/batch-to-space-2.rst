BatchToSpace
============


.. meta::
  :description: Learn about BatchToSpace-2 - a data movement operation,
                which can be performed on four required input tensors.

**Versioned name**: *BatchToSpace-2*

**Category**: *Data movement*

**Short description**: *BatchToSpace* operation permutes the batch dimension on a given input ``data`` into blocks in the spatial dimensions specified by ``block_shape`` input. The spatial dimensions are then optionally cropped according to ``crops_begin`` and ``crops_end`` inputs to produce the output.

**Detailed description**

*BatchToSpace* operation is equivalent to the following operation steps on the input ``data`` with shape ``[batch, D_1, D_2, ..., D_{N-1}]`` and ``block_shape``, ``crops_begin``, ``crops_end`` inputs with shape ``[N]`` to produce the output tensor :math:`y`.

1. Reshape ``data`` input to produce a tensor of shape :math:`[B_1, \dots, B_{N - 1}, \frac{batch}{\left(B_1 \times \dots \times B_{N - 1}\right)}, D_1, D_2, \dots, D_{N - 1}]`

.. math::

   x^{\prime} = reshape(data, [B_1, \dots, B_{N - 1}, \frac{batch}{\left(B_1 \times \dots \times B_{N - 1}\right)}, D_1, D_2, \dots, D_{N - 1}])

2. Permute dimensions of :math:`x^{\prime}` to produce a tensor of shape :math:`[\frac{batch}{\left(B_1 \times \dots \times B_{N - 1}\right)}, D_1, B_1, D_2, B_2, \dots, D_{N-1}, B_{N - 1}]`

.. math::

   x^{\prime\prime} = transpose(x', [N, N + 1, 0, N + 2, 1, \dots, N + N - 1, N - 1])

3. Reshape :math:`x^{\prime\prime}` to produce a tensor of shape :math:`[\frac{batch}{\left(B_1 \times \dots \times B_{N - 1}\right)}, D_1 \times B_1, D_2 \times B_2, \dots, D_{N - 1} \times B_{N - 1}]`

.. math::

   x^{\prime\prime\prime} = reshape(x^{\prime\prime}, [\frac{batch}{\left(B_1 \times \dots \times B_{N - 1}\right)}, D_1 \times B_1, D_2 \times B_2, \dots, D_{N - 1} \times B_{N - 1}])

4. Crop the start and end of spatial dimensions of :math:`x^{\prime\prime\prime}` according to ``crops_begin`` and ``crops_end`` inputs to produce the output :math:`y` of shape:

.. math::

   \left[\frac{batch}{\left(B_1 \times \dots \times B_{N - 1}\right)}, crop(D_1 \times B_1, CB_1, CE_1), crop(D_2 \times B_2, CB_2, CE_2), \dots , crop(D_{N - 1} \times B_{N - 1}, CB_{N - 1}, CE_{N - 1})\right]

Where

- :math:`B_i` = block_shape[i]
- :math:`B_0` is expected to be 1
- :math:`CB_i` = crops_begin[i]
- :math:`CE_i` = crops_end[i]
- :math:`CB_0` and :math:`CE_0` are expected to be 0
- :math:`CB_i + CE_i \leq D_i \times B_i`

*BatchToSpace* operation is the reverse of *SpaceToBatch* operation.

**Attributes**: *BatchToSpace* operation has no attributes.

**Inputs**

*   **1**: ``data`` - A tensor of type *T* and rank greater than or equal to 2. Layout is ``[batch, D_1, D_2 ... D_{N-1}]`` (number of batches, spatial axes). **Required.**
*   **2**: ``block_shape`` - Specifies the block sizes of ``batch`` axis of ``data`` input which are moved to the corresponding spatial axes. A 1D tensor of type *T_INT* and shape ``[N]``. All element values must be greater than or equal to 1. ``block_shape[0]`` is expected to be 1. **Required.**
*   **3**: ``crops_begin`` - Specifies the amount to crop from the beginning along each axis of ``data`` input. A 1D tensor of type *T_INT* and shape ``[N]``. All element values must be greater than or equal to 0. ``crops_begin[0]`` is expected to be 0. **Required.**
*   **4**: ``crops_end`` - Specifies the amount to crop from the ending along each axis of ``data`` input. A 1D tensor of type *T_INT* and shape ``[N]``. All element values must be greater than or equal to 0. ``crops_end[0]`` is expected to be 0. **Required.**
*   **Note**: ``N`` corresponds to the rank of ``data`` input.
*   **Note**: ``batch`` axis of ``data`` input must be evenly divisible by the cumulative product of ``block_shape`` elements.
*   **Note**: It is required that ``crops_begin[i] + crops_end[i] <= block_shape[i] * input_shape[i]``.

**Outputs**

*   **1**: Permuted tensor of type *T* with the same rank as ``data`` input tensor, and shape ``[batch / (block_shape[0] * block_shape[1] * ... * block_shape[N - 1]), D_1 * block_shape[1] - crops_begin[1] - crops_end[1], D_2 * block_shape[2] - crops_begin[2] - crops_end[2], ..., D_{N - 1} * block_shape[N - 1] - crops_begin[N - 1] - crops_end[N - 1]``.

**Types**

* *T*: any supported type.
* *T_INT*: any supported integer type.

**Examples**

Example: 2D input tensor ``data``

.. code-block:: xml
   :force:

   <layer type="BatchToSpace" ...>
       <input>
           <port id="0">       <!-- data -->
               <dim>10</dim>   <!-- batch -->
               <dim>2</dim>    <!-- spatial dimension 1 -->
           </port>
           <port id="1">       <!-- block_shape value: [1, 5] -->
               <dim>2</dim>
           </port>
           <port id="2">       <!-- crops_begin value: [0, 2] -->
               <dim>2</dim>
           </port>
           <port id="3">       <!-- crops_end value: [0, 0] -->
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="3">
               <dim>2</dim>    <!-- data.shape[0] / (block_shape.shape[0] * block_shape.shape[1]) -->
               <dim>8</dim>    <!-- data.shape[1] * block_shape.shape[1] - crops_begin[1] - crops_end[1]-->
           </port>
       </output>
   </layer>

Example: 5D input tensor ``data``

.. code-block:: xml
   :force:

   <layer type="BatchToSpace" ...>
       <input>
           <port id="0">       <!-- data -->
               <dim>48</dim>   <!-- batch -->
               <dim>3</dim>    <!-- spatial dimension 1 -->
               <dim>3</dim>    <!-- spatial dimension 2 -->
               <dim>1</dim>    <!-- spatial dimension 3 -->
               <dim>3</dim>    <!-- spatial dimension 4 -->
           </port>
           <port id="1">       <!-- block_shape value: [1, 2, 4, 3, 1] -->
               <dim>5</dim>
           </port>
           <port id="2">       <!-- crops_begin value: [0, 0, 1, 0, 0] -->
               <dim>5</dim>
           </port>
           <port id="3">       <!-- crops_end value: [0, 0, 1, 0, 0] -->
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="3">
               <dim>2</dim>    <!-- data.shape[0] / (block_shape.shape[0] * block_shape.shape[1] * ... * block_shape.shape[4]) -->
               <dim>6</dim>    <!-- data.shape[1] * block_shape.shape[1] - crops_begin[1] - crops_end[1]-->
               <dim>10</dim>   <!-- data.shape[2] * block_shape.shape[2] - crops_begin[2] - crops_end[2] -->
               <dim>3</dim>    <!-- data.shape[3] * block_shape.shape[3] - crops_begin[3] - crops_end[3] -->
               <dim>3</dim>    <!-- data.shape[4] * block_shape.shape[4] - crops_begin[4] - crops_end[4] -->
           </port>
       </output>
   </layer>


