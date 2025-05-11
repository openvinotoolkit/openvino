Slice
=====


.. meta::
  :description: Learn about Slice-8 - a data movement operation,
                which can be performed on four required and one optional input tensor.

**Versioned name**: *Slice-8*

**Category**: *Data movement*

**Short description**: *Slice* operation extracts a slice of the input tensor.

**Detailed Description**: *Slice* operation selects a region of values from the ``data`` tensor.
Selected values start at indexes provided in the ``start`` input (inclusively) and end
at indexes provides in ``stop`` input (exclusively).

The ``step`` input allows subsampling of ``data``, selecting every *n*-th element,
where ``n`` is equal to ``step`` element for corresponding axis.
Negative ``step`` value indicates slicing backwards, so the sequence along the corresponding axis is reversed in the output tensor.
To select all values contiguously set ``step`` to ``1`` for each axis.

The optional ``axes`` input allows specifying slice indexes only on selected axes.
Other axes will not be affected and will be output in full.

The rules follow python language slicing ``data[start:stop:step]``.

**Attributes**: *Slice* operation has no attributes.

**Inputs**

* **1**: ``data`` - tensor (to be sliced) of type *T* and shape rank greater or equal to 1. **Required.**

* **2**: ``start`` - 1D tensor of type *T_IND*. Indices corresponding to axes in ``data``.

  Defines the starting coordinate of the slice in the ``data`` tensor.
  A negative index value represents counting elements from the end of that dimension.
  A value larger than the size of a dimension is silently clamped. **Required.**

* **3**: ``stop`` - 1D, type *T_IND*, similar to ``start``.

  Defines the coordinate of the opposite vertex of the slice, or where the slice ends.
  Stop indexes are exclusive, which means values lying on the ending edge are
  not included in the output slice.
  To slice to the end of a dimension of unknown size ``INT_MAX``
  may be used (or ``INT_MIN`` if slicing backwards). **Required.**

* **4**: ``step`` - 1D tensor of type *T_IND* and the same shape as ``start`` and ``stop``.

  Integer value that specifies the increment between each index used in slicing.
  Value cannot be ``0``, negative value indicates slicing backwards. **Required.**

* **5**: ``axes`` - 1D tensor of type *T_AXIS*.

  Optional 1D tensor indicating which dimensions the values in ``start`` and ``stop`` apply to.
  Negative value means counting dimensions from the end. The range is ``[-r, r - 1]``, where ``r`` is the rank of the ``data`` input tensor.
  Values are required to be unique. If a particular axis is unspecified, it will be output in full and not sliced.
  Default value: ``[0, 1, 2, ..., start.shape[0] - 1]``. **Optional.**

Number of elements in ``start``, ``stop``, ``step``, and ``axes`` inputs are required to be equal.

**Outputs**

* **1**: Tensor of type *T* with values of the selected slice. The shape of the output tensor has the same rank as the shape of ``data`` input and reduced dimensions according to the values specified by ``start``, ``stop``, and ``step`` inputs.

**Types**

* *T*: any arbitrary supported type.
* *T_IND*: any supported integer type.
* *T_AXIS*: any supported integer type.


**Examples**

Example 1: basic slicing

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [1] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [8] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [1] -->
             <dim>1</dim>
           </port>
           <port id="4">       <!-- axes: [0] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output: [1, 2, 3, 4, 5, 6, 7] -->
               <dim>7</dim>
           </port>
       </output>
   </layer>


Example 2: basic slicing, ``axes`` default

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [1] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [8] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [1] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="4">       <!-- output: [1, 2, 3, 4, 5, 6, 7] -->
               <dim>7</dim>
           </port>
       </output>
   </layer>


Example 3: basic slicing, ``step: [2]``

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [1] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [8] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [2] -->
             <dim>1</dim>
           </port>
           <port id="4">       <!-- axes: [0] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output: [1, 3, 5, 7] -->
               <dim>4</dim>
           </port>
       </output>
   </layer>

Example 4: ``start`` and ``stop`` out of the dimension size, ``step: [1]``

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [-100] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [100] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [1] -->
             <dim>1</dim>
           </port>
           <port id="4">       <!-- axes: [0] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
               <dim>10</dim>
           </port>
       </output>
   </layer>


Example 5: slicing backward all elements, ``step: [-1]``, ``stop: [-11]``

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [9] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [-11] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [-1] -->
             <dim>1</dim>
           </port>
           <port id="4">       <!-- axes: [0] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -->
               <dim>10</dim>
           </port>
       </output>
   </layer>


Example 6: slicing backward, ``step: [-1]``, ``stop: [0]``

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [9] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [0] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [-1] -->
             <dim>1</dim>
           </port>
           <port id="4">       <!-- axes: [0] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output: [9, 8, 7, 6, 5, 4, 3, 2, 1] -->
               <dim>9</dim>
           </port>
       </output>
   </layer>


Example 7: slicing backward, ``step: [-1]``, ``stop: [-10]``

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [9] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [-10] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [-1] -->
             <dim>1</dim>
           </port>
           <port id="4">       <!-- axes: [0] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output: [9, 8, 7, 6, 5, 4, 3, 2, 1] -->
               <dim>9</dim>
           </port>
       </output>
   </layer>


Example 8: slicing backward, ``step: [-2]``

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [9] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [-11] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [-2] -->
             <dim>1</dim>
           </port>
           <port id="4">       <!-- axes: [0] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output: [9, 7, 5, 3, 1] -->
               <dim>5</dim>
           </port>
       </output>
   </layer>


Example 9: ``start`` and ``stop`` out of the dimension size, slicing backward

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -->
             <dim>10</dim>
           </port>
           <port id="1">       <!-- start: [100] -->
             <dim>1</dim>
           </port>
           <port id="2">       <!-- stop: [-100] -->
             <dim>1</dim>
           </port>
           <port id="3">       <!-- step: [-1] -->
             <dim>1</dim>
           </port>
           <port id="4">       <!-- axes: [0] -->
             <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -->
               <dim>10</dim>
           </port>
       </output>
   </layer>


Example 10: slicing 2D tensor, all axes specified

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data: data: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]] -->
             <dim>2</dim>
             <dim>5</dim>
           </port>
           <port id="1">       <!-- start: [0, 1] -->
             <dim>2</dim>
           </port>
           <port id="2">       <!-- stop: [2, 4] -->
             <dim>2</dim>
           </port>
           <port id="3">       <!-- step: [1, 2] -->
             <dim>2</dim>
           </port>
           <port id="4">       <!-- axes: [0, 1] -->
             <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="5">      <!-- output: [1, 3, 6, 8] -->
               <dim>2</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>


Example 11: slicing 3D tensor, all axes specified

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data -->
             <dim>20</dim>
             <dim>10</dim>
             <dim>5</dim>
           </port>
           <port id="1">       <!-- start: [0, 0, 0] -->
             <dim>2</dim>
           </port>
           <port id="2">       <!-- stop: [4, 10, 5] -->
             <dim>2</dim>
           </port>
           <port id="3">       <!-- step: [1, 1, 1] -->
             <dim>2</dim>
           </port>
           <port id="4">       <!-- axes: [0, 1, 2] -->
             <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output -->
               <dim>4</dim>
               <dim>10</dim>
               <dim>5</dim>
           </port>
       </output>
   </layer>

Example 12: slicing 3D tensor, last axes default

.. code-block:: xml
   :force:

   <layer id="1" type="Slice" ...>
       <input>
           <port id="0">       <!-- data -->
             <dim>20</dim>
             <dim>10</dim>
             <dim>5</dim>
           </port>
           <port id="1">       <!-- start: [0, 0] -->
             <dim>2</dim>
           </port>
           <port id="2">       <!-- stop: [4, 10] -->
             <dim>2</dim>
           </port>
           <port id="3">       <!-- step: [1, 1] -->
             <dim>2</dim>
           </port>
           <port id="4">       <!-- axes: [0, 1] -->
             <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="5">       <!-- output -->
               <dim>4</dim>
               <dim>10</dim>
               <dim>5</dim>
           </port>
       </output>
   </layer>

