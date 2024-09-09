ConvertLike
===========


.. meta::
  :description: Learn about ConvertLike-1 - an element-wise, type conversion
                operation, which can be performed two required input tensors.

**Versioned name**: *ConvertLike-1*

**Category**: *Type conversion*

**Short description**: *ConvertLike* operation performs element-wise conversion on a given input tensor ``data`` to the element type of an additional input tensor ``like``.

**Detailed description**

Conversion from one supported type to another supported type is always allowed. User must be aware of precision loss and value change caused by range difference between two types. For example, a 32-bit float *3.141592* may be round to a 32-bit int *3*. The result of unsupported conversions is undefined, e.g. conversion of negative signed integer value to any unsigned integer type.

Output elements are represented as follows:

.. code-block:: cpp

    o[i] = Convert[destination_type=type(b)](a[i])

where ``a`` and ``b`` correspond to ``data`` and ``like`` input tensors, respectively.

**Attributes**: *ConvertLike* operation has no attributes.

**Inputs**

* **1**: ``data`` - A tensor of type *T1* and arbitrary shape. **Required.**
* **2**: ``like`` - A tensor of type *T2* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *ConvertLike* operation applied to input tensor ``data``. A tensor of type *T2* and the same shape as ``data`` input tensor.

**Types**

* *T1*: any supported type
* *T2*: any supported type

**Example**

.. code-block:: cpp

   <layer ... type="ConvertLike">
       <input>
           <port id="0">        <!-- type: int32 -->
               <dim>256</dim>
               <dim>56</dim>
           </port>
           <port id="1">        <!-- type: float32 -->
               <dim>3</dim>     <!-- any data -->
           </port>
       </input>
       <output>
           <port id="2">        <!-- result type: float32 -->
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </output>
   </layer>


