IsNaN
=====


.. meta::
  :description: Learn about IsNaN-10 - an element-wise, comparison operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *IsNaN-10*

**Category**: *Comparison*

**Short description**: *IsNaN* returns the boolean mask of a given tensor which maps ``NaN`` to ``True``.

**Detailed description**: *IsNaN* returns the boolean mask of the input tensor in which ``True`` corresponds to ``NaN`` and ``False`` to other values.
The output tensor has the same shape as the input tensor.
The ``i``'th element of the output tensor is ``True`` if  ``i``'th element of the input tensor is ``NaN``. Otherwise, it is ``False``.
For example, for the given input tensor ``[NaN, 2.1, 3.7, NaN, Inf]`` the output tensor is ``[True, False, False, True, False]``.

**Attributes**

*IsNaN* operation has no attributes.

**Inputs**:

* **1**: A tensor of type *T_IN* and arbitrary shape. **Required.**

**Outputs**:

* **1**: A ``boolean`` tensor of the same shape as the input tensor.

**Types**

* *T_IN*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="IsNaN">
       <input>
           <port id="0" precision="FP32">
               <dim>256</dim>
               <dim>56</dim>
           </port>
        </input>
       <output>
           <port id="1" precision="BOOL">
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </output>
   </layer>


