Cos
===


.. meta::
  :description: Learn about Cos-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Cos-1*

**Category**: *Arithmetic unary*

**Short description**: *Cos* performs element-wise cosine operation on a given input tensor.

**Detailed description**: *Cos* performs element-wise cosine operation on a given input tensor, based on the following mathematical formula:

.. math::

   a_{i} = cos(a_{i})

**Attributes**: *Cos* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *Cos* operation. A tensor of type *T* and the same shape as the input tensor.

**Types**

* *T*: any numeric type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="Cos">
       <input>
           <port id="0">
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </output>
   </layer>

