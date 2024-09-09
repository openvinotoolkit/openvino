SpaceToBatch
============


.. meta::
  :description: Learn about SpaceToBatch-2 - a data movement operation,
                which can be performed on four required input tensors.

**Versioned name**: *SpaceToBatch-2*

**Category**: *Data movement*

**Short description**: The *SpaceToBatch* operation divides "spatial" dimensions ``[1, ..., N - 1]`` of the ``data`` input into a grid of blocks of shape ``block_shape``, and interleaves these blocks with the batch dimension (0) such that in the output, the spatial dimensions ``[1, ..., N - 1]`` correspond to the position within the grid, and the batch dimension combines both the position within a spatial block and the original batch position. Prior to division into blocks, the spatial dimensions of the input are optionally zero padded according to ``pads_begin`` and ``pads_end``.

**Detailed description**:

The operation is equivalent to the following transformation of the input tensor ``data`` of shape ``[batch, D_1, D_2 ... D_{N - 1}]`` and ``block_shape``, ``pads_begin``, ``pads_end`` of shapes ``[N]`` to *Y* output tensor.

Zero-pad the start and end of dimensions  :math:`[D_0, \dots, D_{N - 1}]` of the input according to ``pads_begin`` and ``pads_end``:

.. math::

	x = [batch + P_0, D_1 + P_1, D_2 + P_2, \dots, D_{N - 1} + P_{N - 1}]



.. math::

	x' = reshape(x, [batch, \frac{D_1 + P_1}{B_1}, B_1, \frac{D_2 + P_2}{B_2}, B_2, \dots, \frac{D_{N - 1} + P_{N - 1}}{B_{N - 1}}, B_{N - 1}])



.. math::

	x'' = transpose(x', [2, 4, \dots, (N - 1) + (N - 1), 0, 1, 3, \dots, N + (N - 1)])



.. math::

	y = reshape(x'', [batch \times B_1 \times \dots \times B_{N - 1}, \frac{D_1 + P_1}{B_1}, \frac{D_2 + P_2}{B_2}, \dots, \frac{D_{N - 1} + P_{N - 1}}{B_{N - 1}}]

where

* :math:`P_i` = pads_begin[i] + pads_end[i]

* :math:`B_i` = block_shape[i]

* :math:`P_0` for batch dimension is expected to be 0 (no-padding)

* :math:`B_0` for batch is ignored

**Attributes**

No attributes available.

**Inputs**

*   **1**: ``data`` - input N-D tensor ``[batch, D_1, D_2 ... D_{N - 1}]`` of *T1* type with rank >= 2. **Required.**
*   **2**: ``block_shape`` - input 1-D tensor of *T2* type with shape ``[N]`` that is equal to the size of ``data`` input shape. All values must be >= 1.  ``block_shape[0]`` is expected to be 1. **Required.**
*   **3**: ``pads_begin`` - input 1-D tensor of *T2* type with shape ``[N]`` that is equal to the size of ``data`` input shape. All values must be non-negative. ``pads_begin`` specifies the padding for the beginning along each axis of ``data`` input . It is required that ``block_shape[i]`` divides ``data_shape[i] + pads_begin[i] + pads_end[i]``. ``pads_begin[0]`` is expected to be 0. **Required.**
*   **4**: ``pads_end`` - input 1-D tensor of *T2* type with shape ``[N]`` that is equal to the size of ``data`` input shape. All values must be non-negative. ``pads_end`` specifies the padding for the ending along each axis of ``data`` input. It is required that ``block_shape[i]`` divides ``data_shape[i] + pads_begin[i] + pads_end[i]``. ``pads_end[0]`` is expected to be 0. **Required.**

**Outputs**

*   **1**: N-D tensor with shape ``[batch * block_shape[0] * block_shape[1] * ... * block_shape[N - 1], (D_1 + pads_begin[1] + pads_end[1]) / block_shape[1], (D_2 + pads_begin[2] + pads_end[2]) / block_shape[2], ..., (D_{N -1} + pads_begin[N - 1] + pads_end[N - 1]) / block_shape[N - 1]`` of the same type as ``data`` input.

**Types**

* *T1*: any supported type.
* *T2*: any supported integer type.

**Example**

.. code-block:: xml
   :force:

    <layer type="SpaceToBatch" ...>
        <input>
            <port id="0">       <!-- data -->
                <dim>2</dim>    <!-- batch -->
                <dim>6</dim>    <!-- spatial dimension 1 -->
                <dim>10</dim>   <!-- spatial dimension 2 -->
                <dim>3</dim>    <!-- spatial dimension 3 -->
                <dim>3</dim>    <!-- spatial dimension 4 -->
            </port>
            <port id="1">       <!-- block_shape value: [1, 2, 4, 3, 1] -->
                <dim>5</dim>
            </port>
            <port id="2">       <!-- pads_begin value: [0, 0, 1, 0, 0] -->
                <dim>5</dim>
            </port>
            <port id="3">       <!-- pads_end value: [0, 0, 1, 0, 0] -->
                <dim>5</dim>
            </port>
        </input>
        <output>
            <port id="3">
                <dim>48</dim>   <!-- data.shape[0] * block_shape.shape[0] * block_shape.shape[1] *... * block_shape.shape[4] -->
                <dim>3</dim>    <!-- (data.shape[1] + pads_begin[1] + pads_end[1]) / block_shape.shape[1]  -->
                <dim>3</dim>    <!-- (data.shape[2] + pads_begin[2] + pads_end[2]) / block_shape.shape[2] -->
                <dim>1</dim>    <!-- (data.shape[3] + pads_begin[3] + pads_end[3]) / block_shape.shape[3] -->
                <dim>3</dim>    <!-- (data.shape[4] + pads_begin[4] + pads_end[4]) / block_shape.shape[4] -->
            </port>
        </output>
    </layer>

