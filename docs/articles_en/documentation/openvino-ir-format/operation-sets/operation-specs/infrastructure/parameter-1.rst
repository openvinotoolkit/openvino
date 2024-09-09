Parameter
=========


.. meta::
  :description: Learn about Parameter-1 - an infrastructure operation, which
                can be performed to specify input to the model.

**Versioned name**: *Parameter-1*

**Category**: *Infrastructure*

**Short description**: *Parameter* layer specifies input to the model.

**Attributes**:

* *element_type*

  * **Description**: the type of element of output tensor
  * **Range of values**: u1, u4, u8, u16, u32, u64, i4, i8, i16, i32, i64, f16, f32, boolean, bf16
  * **Type**: ``string``
  * **Required**: *yes*

* *shape*

  * **Description**: the shape of the output tensor
  * **Range of values**: list of non-negative integers, empty list is allowed, which means 0D or scalar tensor
  * **Type**: ``int[]``
  * **Required**: *yes*


**Outputs**

* **1**: Output tensor of type *T* and shape equal to *shape* attribute.

**Types**

* *T*: any type from *element type* values.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="Parameter" ...>
       <data>element_type="f32" shape="1,3,224,224"</data>
       <output>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
       </output>
   </layer>


