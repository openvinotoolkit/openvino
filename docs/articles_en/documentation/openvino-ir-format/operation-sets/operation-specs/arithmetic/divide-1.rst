Divide
======


.. meta::
  :description: Learn about Divide-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Divide-1*

**Category**: *Arithmetic binary*

**Short description**: *Divide* performs element-wise division operation with two given tensors applying broadcasting rule specified in the *auto_broadcast* attribute.

**Detailed description**
Before performing arithmetic operation, input tensors *a* and *b* are broadcasted if their shapes are different and ``auto_broadcast`` attribute is not ``none``. Broadcasting is performed according to ``auto_broadcast`` value.
After broadcasting *Divide* performs division operation for the input tensors *a* and *b* using the formula below:

.. math::

   o_{i} = \frac{a_{i}}{b_{i}}

The result of division by zero is undefined.

**Attributes**:

* *m_pythondiv*

  * **Description**: specifies if floor division should be calculate. This attribute is supported only for integer data types.
  * **Range of values**:

    * false - regular division
    * true - floor division
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes must match,
    * *numpy* - numpy broadcasting rules, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`,
    * *pdpd* - PaddlePaddle-style implicit broadcasting, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`.
  * **Type**: ``string``
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape and rank. **Required.**
* **2**: A tensor of type *T* and arbitrary shape and rank. **Required.**

**Outputs**

* **1**: The result of element-wise division operation. A tensor of type *T* with shape equal to broadcasted shape of the two inputs.

**Types**

* *T*: any numeric type.

**Examples**

*Example 1*

.. code-block:: xml
   :force:

   <layer ... type="Divide">
       <data auto_broadcast="none" m_pythondiv="true"/>
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

*Example 2: broadcast*

.. code-block:: xml
   :force:

   <layer ... type="Divide">
       <data auto_broadcast="numpy" m_pythondiv="false"/>
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


