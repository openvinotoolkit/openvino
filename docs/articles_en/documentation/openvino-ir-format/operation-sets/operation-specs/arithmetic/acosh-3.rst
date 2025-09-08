Acosh
=====


.. meta::
  :description: Learn about Acosh-3 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Acosh-3*

**Category**: *Arithmetic unary*

**Short description**: *Acosh* performs element-wise hyperbolic inverse cosine (arccosh) operation with given tensor.

**Detailed description**:  Operation takes one input tensor and performs the element-wise hyperbolic inverse cosine operation on a given input tensor, based on the following mathematical formula:

.. math::

   a_{i} = acosh(a_{i})

**Attributes**: *Acosh* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *Acosh* operation. A tensor of type *T* and the same shape as the input tensor.

**Types**

* *T*: any numeric type.

**Examples**

.. code-block:: xml
   :force:

   <layer ... type="Acosh">
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


