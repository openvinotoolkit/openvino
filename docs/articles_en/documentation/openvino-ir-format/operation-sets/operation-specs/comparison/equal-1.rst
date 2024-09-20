Equal
=====


.. meta::
  :description: Learn about Equal-1 - an element-wise, comparison operation, which
                can be performed on two given tensors in OpenVINO.

**Versioned name**: *Equal-1*

**Category**: *Comparison binary*

**Short description**: *Equal* performs element-wise comparison operation with two given input tensors applying multi-directional broadcast rules specified in the *auto_broadcast* attribute.

**Detailed description**
Before performing arithmetic operation, input tensors *a* and *b* are broadcasted if their shapes are different and *auto_broadcast* attributes is not *none*. Broadcasting is performed according to *auto_broadcast* value.

After broadcasting *Equal* does the following with the input tensors *a* and *b*:

.. math::

   o_{i} = a_{i} == b_{i}

**Attributes**:

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes should match,
    * *numpy* - numpy broadcasting rules, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`,
    * *pdpd* - PaddlePaddle-style implicit broadcasting, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`.
  * **Type**: ``string``
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**
* **2**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise **comparison** operation applied to the input tensors. A tensor of type *T_BOOL* and the same shape equal to broadcasted shape of two inputs.

**Types**

* *T*: arbitrary supported type.
* *T_BOOL*: ``boolean``.

**Examples**

*Example 1: no broadcast*

.. code-block:: xml
   :force:

   <layer ... type="Equal">
       <data auto_broadcast="none"/>
       <input>
           <port id="0">
               <dim>256</dim>
               <dim>56</dim>
           </port>
           <port id="1">
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </output>
   </layer>

*Example 2: numpy broadcast*

.. code-block:: xml
   :force:

   <layer ... type="Equal">
       <data auto_broadcast="numpy"/>
       <input>
           <port id="0">
               <dim>8</dim>
               <dim>1</dim>
               <dim>6</dim>
               <dim>1</dim>
           </port>
           <port id="1">
               <dim>7</dim>
               <dim>1</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>8</dim>
               <dim>7</dim>
               <dim>6</dim>
               <dim>5</dim>
           </port>
       </output>
   </layer>


