Eye
===


.. meta::
  :description: Learn about Eye-9 - a generation operation, which can be
                performed on three required and one optional input tensors.

**Versioned name**: *Eye-9*

**Category**: *Generation*

**Short description**: *Eye* operation generates shift matrix or a batch of matrices.

**Detailed description**:

*Eye* operation generates an identity matrix or a batch matrices with ones on the diagonal and zeros everywhere else. The index of the diagonal to be populated with ones is given by ``diagonal_index``: ``output[..., i, i + diagonal_index] = 1``.


Example 1. *Eye* output with ``output_type`` = ``i32``:

.. code-block:: xml
   :force:

   num_rows = 3

   num_columns = 4

   diagonal_index = 2

   output  = [[0 0 1 0]
              [0 0 0 1]
              [0 0 0 0]]

Example 2. *Eye* output with ``output_type`` = ``i32``:

.. code-block:: xml
   :force:

   num_rows = 3

   num_columns = 4

   diagonal_index = -1

   output  = [[0 0 0 0]
              [1 0 0 0]
              [0 1 0 0]]

Example 3. *Eye* output with ``output_type`` = ``f16``:

.. code-block:: xml
   :force:

   num_rows = 2

   diagonal_index = 5

   batch_shape = [1, 2]

   output  = [[[[0. 0.]
                [0. 0.]]
               [[0. 0.]
                [0. 0.]]]]

**Attributes**:

* *output_type*

  * **Description**: the type of the output
  * **Range of values**: any numeric type
  * **Type**: ``string``
  * **Required**: *Yes*


**Inputs**:

* **1**: ``num_rows`` - scalar or 1D tensor with 1 non-negative element of type *T_NUM* describing the number of rows in matrix. **Required.**
* **2**: ``num_columns`` - scalar or 1D tensor with 1 non-negative element of type *T_NUM* describing the number of columns in matrix. **Required.**
* **3**: ``diagonal_index`` - scalar or 1D tensor with element of type *T_NUM* describing the index of the diagonal to be populated. A positive value refers to an upper diagonal and a negative value refers to a lower diagonal. Value ``0`` populates the main diagonal. If ``diagonal_index`` is a positive value and is not smaller than ``num_rows`` or if ``diagonal_index`` is a negative value and is not larger than ``num_columns``, the matrix will be filled with only zeros. **Required.**
* **4**: ``batch_shape`` - 1D tensor with non-negative values of type *T_NUM* defines leading batch dimensions of output shape. If ``batch_shape`` is an empty list, *Eye* operation generates a 2D tensor (matrix). This input is optional, and its default value equal to an empty tensor.


**Outputs**:

* **1**: A tensor with the type specified by the *output_type* attribute. The shape is ``batch_shape + [num_rows, num_columns]``

**Types**

* *T_NUM*: ``int32`` or ``int64``.

**Examples**

*Example 1*

.. code-block:: xml
   :force:

   <layer ... name="Eye" type="Eye">
       <data output_type="i8"/>
       <input>
           <port id="0" precision="I32"/>  <!-- num rows: 5 -->
           <port id="1" precision="I32"/>  <!-- num columns: 5 -->
           <port id="2" precision="I32"/>  <!-- diagonal index -->
       </input>
       <output>
           <port id="3" precision="I8" names="Eye:0">
               <dim>5</dim>
               <dim>5</dim>
           </port>
       </output>
   </layer>

*Example 2*

.. code-block:: xml
   :force:

   <layer ... name="Eye" type="Eye">
       <data output_type="f32"/>
       <input>
           <port id="0" precision="I32"/>  <!-- num rows -->
           <port id="1" precision="I32"/>  <!-- num columns -->
           <port id="2" precision="I32"/>  <!-- diagonal index -->
           <port id="3" precision="I32"/>  <!-- batch_shape : [2, 3] -->
       </input>
       <output>
           <port id="3" precision="F32" names="Eye:0">
               <dim>2</dim>
               <dim>3</dim>
               <dim>-1</dim>
               <dim>-1</dim>
           </port>
       </output>
   </layer>


