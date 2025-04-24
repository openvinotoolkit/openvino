Round
=====


.. meta::
  :description: Learn about Round-5 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Round-5*

**Category**: *Arithmetic unary*

**Short description**: *Round* performs element-wise round operation with given tensor.

**Detailed description**: Operation takes one input tensor and rounds the values, element-wise, meaning it finds the nearest integer for each value. In case of halves, the rule is to round them to the nearest even integer if ``mode`` attribute is ``half_to_even`` or rounding in such a way that the result heads away from zero if ``mode`` attribute is ``half_away_from_zero``.

.. code-block:: xml
   :force:

   Input = [-4.5, -1.9, -1.5, 0.5, 0.9, 1.5, 2.3, 2.5]

   round(Input, mode = `half_to_even`) = [-4.0, -2.0, -2.0, 0.0, 1.0, 2.0, 2.0, 2.0]

   round(Input, mode = `half_away_from_zero`) = [-5.0, -2.0, -2.0, 1.0, 1.0, 2.0, 2.0, 3.0]

**Attributes**:

* *mode*

  * **Description**:  If set to ``half_to_even`` then the rule is to round halves to the nearest even integer, if set to ``half_away_from_zero`` then rounding in such a way that the result heads away from zero.
  * **Range of values**: ``half_to_even`` or ``half_away_from_zero``
  * **Type**: string
  * **Default value**: ``half_to_even``
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise round operation. A tensor of type *T*.

**Types**

* *T*: any numeric type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="Round">
       <data mode="half_to_even"/>
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

