GatherElements
==============



.. meta::
  :description: Learn about GatherElements-6 - a data movement operation,
                which can be performed on two required input tensors.

**Versioned name**: *GatherElements-6*

**Category**: *Data movement*

**Short description**: *GatherElements* takes elements from the input ``data`` tensor at positions specified in the ``indices`` tensor.

**Detailed description** *GatherElements* takes elements from the ``data`` tensor at positions specified
in the ``indices`` tensor. The ``data`` and ``indices`` tensors have the same rank ``r >= 1``. Optional
attribute ``axis`` determines along which axis elements with indices specified in ``indices`` are taken.
The ``indices`` tensor has the same shape as the ``data`` tensor except for the ``axis`` dimension.
Output consists of values (gathered from the ``data`` tensor) for each element in the ``indices`` tensor
and has the same shape as ``indices``.

For instance, in the 3D case (``r = 3``), the output is determined by the following equations:

.. code-block:: sh

   out[i][j][k] = data[indices[i][j][k]][j][k] if axis = 0
   out[i][j][k] = data[i][indices[i][j][k]][k] if axis = 1
   out[i][j][k] = data[i][j][indices[i][j][k]] if axis = 2

Example 1 with concrete values:

.. code-block:: sh

   data = [
       [1, 2],
       [3, 4],
   ]
   indices = [
       [0, 1],
       [0, 0],
   ]
   axis = 0
   output = [
       [1, 4],
       [1, 2],
   ]

Example 2 with ``axis`` = 1 and ``indices`` having greater (than ``data``) shape:

.. code-block:: sh

   data = [
       [1, 7],
       [4, 3],
   ]
   indices = [
       [1, 1, 0],
       [1, 0, 1],
   ]
   axis = 1
   output = [
       [7, 7, 1],
       [3, 4, 3],
   ]


Example 3 ``indices`` has lesser (than ``data``) shape:

.. code-block:: sh

   data = [
       [1, 2, 3],
       [4, 5, 6],
       [7, 8, 9],
   ]
   indices = [
       [1, 0, 1],
       [1, 2, 0],
   ]
   axis = 0
   output = [
       [4, 2, 6],
       [4, 8, 3],
   ]


**Attributes**:

* *axis*

  * **Description**: Which axis to gather on. Negative value means counting dimensions from the back.
  * **Range of values**: ``[-r, r-1]`` where ``r = rank(data)``.
  * **Type**: int
  * **Required**: *yes*


**Inputs**:

* **1**:  Tensor of type *T*. This is a tensor of a ``rank >= 1``. **Required.**
* **2**:  Tensor of type *T_IND* with the same rank as the input. All index values are expected to be
  within bounds ``[0, s-1]``, where ``s`` is size along ``axis`` dimension of the ``data`` tensor. **Required.**

**Outputs**:

* **1**: Tensor with gathered values of type *T*. Tensor has the same shape as ``indices``.

**Types**

* *T*: any supported type.
* *T_IND*: ``int32`` or ``int64``.

**Example**

.. code-block:: xml
   :force:

   <... type="GatherElements" ...>
       <data axis="1" />
       <input>
           <port id="0">
               <dim>3</dim>
               <dim>7</dim>
               <dim>5</dim>
           </port>
           <port id="1">
               <dim>3</dim>
               <dim>10</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>3</dim>
               <dim>10</dim>
               <dim>5</dim>
           </port>
       </output>
   </layer>




