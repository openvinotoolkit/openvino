IsFinite
========


.. meta::
  :description: Learn about IsFinite-10 - an element-wise, comparison operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *IsFinite-10*

**Category**: *Comparison*

**Short description**: *IsFinite* performs element-wise test for finiteness (not Infinity and not NaN) on elements of a given input tensor. It returns the boolean mask of a given tensor which maps
``NaN`` and ``Infinity`` to ``False`` and all other values to ``True``
*IsFinite* operation has no attributes.

**Detailed description**: *IsFinite* returns the boolean mask of the input tensor in which ``False`` corresponds to ``NaN`` and ``Infinity`` and ``True`` to all other values.

* The output tensor has the same shape as the input tensor.
* The ``i``'th element of the output tensor is ``False`` if ``i``'th element of the input tensor is ``NaN`` or ``Inf``. Otherwise, it is ``True``.
* For example, for a given input tensor ``[NaN, 2.1, 3.7, Inf]`` the output tensor is ``[False, True, True, False]``.

**Attributes**

*IsFinite* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *IsFinite* operation. A tensor of type ``boolean`` and the same shape as input tensor.

**Types**

* **T**: any supported floating-point type.

**Examples**

.. code-block:: xml
   :force:

   <layer ... type="IsFinite">
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


.. code-block:: xml
   :force:

   <layer ... type="IsFinite">
       <input>
           <port id="0" precision="FP32">
               <dim>4</dim> <!-- Input value is: [NaN, 2.1, 3.7, Inf] -->
           </port>
       </input>
       <output>
           <port id="1" precision="BOOL">
               <dim>4</dim> <!-- Output value is: [False, True, True, False] -->
           </port>
       </output>
   </layer>



