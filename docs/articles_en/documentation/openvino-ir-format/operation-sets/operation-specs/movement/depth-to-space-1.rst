DepthToSpace
============


.. meta::
  :description: Learn about DepthToSpace-1 - a data movement operation,
                which can be performed on a single input tensor.

**Versioned name**: *DepthToSpace-1*

**Category**: *Data movement*

**Short description**: *DepthToSpace* operation rearranges data from the depth dimension
of the input tensor into spatial dimensions of the output tensor.

**Detailed description**

*DepthToSpace* operation permutes elements from the input tensor with shape ``[N, C, D1,
D2, ..., DK]``, to the output tensor where values from the input depth dimension
(features) ``C`` are moved to spatial blocks in ``D1``, ..., ``DK``.

The operation is equivalent to the following transformation of the input tensor ``data``
with ``K`` spatial dimensions of shape ``[N, C, D1, D2, ..., DK]`` to *Y* output tensor.
If ``mode = blocks_first``:

.. code-block:: py

   x' = reshape(data, [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2, ..., DK])
   x'' = transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K])
   y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size, ..., DK * block_size])

If ``mode = depth_first``:

.. code-block:: py

   x' = reshape(data, [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2, ..., DK])
   x'' = transpose(x', [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1])
   y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size, ..., DK * block_size])

**Attributes**

* *block_size*

  * **Description**: specifies the size of the value block to be moved. The depth dimension size must be evenly divided by ``block_size ^ (len(input.shape) - 2)``.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

* *mode*

  * **Description**: specifies how the input depth dimension is split to block coordinates and the new depth dimension.
  * **Range of values**:

    * *blocks_first*: the input depth is divided to ``[block_size, ..., block_size,  new_depth]``
    * *depth_first*: the input depth is divided to ``[new_depth, block_size, ..., block_size]``
  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**

* **1**: ``data`` - input tensor of type *T* with rank >= 3. **Required.**

**Outputs**

* **1**: permuted tensor of type *T* and shape ``[N, C / block_size ^ K, D1 * block_size, D2 * block_size, ..., DK * block_size]``.

**Types**

* *T*: any supported type.

**Example**

.. code-block:: xml
   :force:

   <layer type="DepthToSpace" ...>
       <data block_size="2" mode="blocks_first"/>
       <input>
           <port id="0">
               <dim>5</dim>
               <dim>28</dim>
               <dim>2</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>5</dim>  <!-- data.shape[0] -->
               <dim>7</dim>  <!-- data.shape[1] / (block_size ^ 2) -->
               <dim>4</dim>  <!-- data.shape[2] * block_size -->
               <dim>6</dim>  <!-- data.shape[3] * block_size -->
           </port>
       </output>
   </layer>

