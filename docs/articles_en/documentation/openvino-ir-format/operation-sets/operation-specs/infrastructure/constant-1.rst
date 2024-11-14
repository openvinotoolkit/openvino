Constant
========


.. meta::
  :description: Learn about Constant-1 - an infrastructure operation, which
                can be performed to produce a tensor with content read from
                binary file by offset and size.

**Versioned name**: *Constant-1*

**Category**: *Infrastructure*

**Short description**: *Constant* operation produces a tensor with content read from binary file by offset and size.

**Attributes**

* *offset*

  * **Description**: specifies position in binary file with weights where the content of the constant begins; value in bytes
  * **Range of values**: non-negative integer value
  * **Type**: ``int``
  * **Required**: *yes*

* *size*

  * **Description**: size of constant content in binary files; value in bytes
  * **Range of values**: positive integer bigger than zero
  * **Type**: ``int``
  * **Required**: *yes*

* *element_type*

  * **Description**: the type of element of output tensor
  * **Range of values**: u1, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, boolean, bf16
  * **Type**: ``string``
  * **Required**: *yes*

* *shape*

  * **Description**: the shape of the output tensor
  * **Range of values**: list of non-negative integers, empty list is allowed, which means 0D or scalar tensor
  * **Type**: ``int[]``
  * **Required**: *yes*


**Outputs**

* **1**: Output tensor of type *T*.

**Types**

* *T*: any type from *element type* values.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="Constant">
       <data offset="1000" size="256" element_type="f32" shape="8,8"/>
       <output>
           <port id="1">
               <dim>8</dim>
               <dim>8</dim>
           </port>
       </output>
   </layer>


