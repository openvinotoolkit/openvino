Asin
====


.. meta::
  :description: Learn about Asin-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Asin-1*

**Category**: *Arithmetic unary*

**Short description**: *Asin* performs element-wise inverse sine (arcsin) operation with given tensor.

**Attributes**:

No attributes available.

**Inputs**

* **1**: An tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise asin operation. A tensor of type *T*.

**Types**

* *T*: any numeric type.

*Asin* does the following with the input tensor *a*:

.. math::

   a_{i} = asin(a_{i})

**Examples**

*Example 1*

.. code-block:: xml
   :force:

   <layer ... type="Asin">
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


