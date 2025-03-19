Convert
=======


.. meta::
  :description: Learn about Convert-1 - an element-wise, type conversion
                operation, which can be performed on a single input tensor.

**Versioned name**: *Convert-1*

**Category**: *Type conversion*

**Short description**: *Convert* operation performs element-wise conversion on a given input tensor to a type specified in the *destination_type* attribute.

**Detailed description**

Conversion from one supported type to another supported type is always allowed. User must be aware of precision loss and value change caused by range difference between two types. For example, a 32-bit float ``3.141592`` may be round to a 32-bit int ``3``.

Conversion of negative signed integer to unsigned integer value happens in accordance with c++ standard. Notably,  result is the unique value of the destination unsigned type that is congruent to the source integer modulo 2^N (where N is the bit width of the destination type). For example, when an int32 value ``-1`` is converted to uint32 the result will be ``uint32 max`` which is ``4,294,967,295``.

The result of unsupported conversions is undefined. Output elements are represented as follows:

.. math::

   o_{i} = Convert(a_{i})

where ``a`` corresponds to the input tensor.

**Attributes**:

* *destination_type*

  * **Description**: the destination type.
  * **Range of values**: one of the supported types *T*
  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *Convert* operation. A tensor of *destination_type* type and the same shape as input tensor.

**Types**

* *T*: any supported type

**Example**

.. code-block:: cpp

   <layer ... type="Convert">
       <data destination_type="f32"/>
       <input>
           <port id="0">        <!-- type: i32 -->
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </input>
       <output>
           <port id="1">        <!-- result type: f32 -->
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </output>
   </layer>


