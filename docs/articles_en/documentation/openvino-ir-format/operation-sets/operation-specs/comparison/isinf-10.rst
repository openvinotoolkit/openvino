IsInf
=====


.. meta::
  :description: Learn about IsInf - an element-wise, comparison operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *IsInf-10*

**Category**: *Comparison*

**Short description**: *IsInf* performs element-wise mapping of infinite values to True.

**Detailed description**: *IsInf* performs element-wise mapping of infinite values to true and other values to false. Attributes ``detect_negative`` and ``detect_positive`` can be used to control the mapping of negative and positive infinities. Setting both ``detect_negative`` and ``detect_positive`` to false will map all values to false.

**Attributes**

* *detect_negative*

  * **Description**: specifies rules used for mapping values with negative infinity.
  * **Range of values**:

    * ``false`` - map negative infinity to ``false``
    * ``true`` - map negative infinity to ``true``
  * **Type**: ``boolean``
  * **Default value**: ``true``
  * **Required**: *no*

* *detect_positive*

  * **Description**: specifies rules used for mapping values with positive infinity.
  * **Range of values**:

    * ``false`` - map positive infinity to ``false``
    * ``true`` - map positive infinity to ``true``
  * **Type**: ``boolean``
  * **Default value**: ``true``
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - Input tensor of type ``T_IN`` with data and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of the element-wise mapping of infinite values applied to the input tensor. A tensor of the ``boolean`` type and shape equal to the input tensor.

**Types**

* **T_IN**: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="IsInf" ...>
       <data detect_negative="true" detect_positive="true"/>
       <input>
           <port id="0" precision="FP32">
               <dim>256</dim>
               <dim>128</dim>
           </port>
       </input>
       <output>
           <port id="0" precision="BOOL">
               <dim>256</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>



