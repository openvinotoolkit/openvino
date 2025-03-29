CumSum
======


.. meta::
  :description: Learn about CumSum-3 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *CumSum-3*

**Category**: *Arithmetic unary*

**Short description**: *CumSum* performs cumulative summation of the input elements along the given axis.

**Detailed description**: *CumSum* performs cumulative summation of the input elements along the ``axis`` specified by the second input. By default, the ``j-th`` output element is the inclusive sum of the first ``j`` elements in the given sequence, and the first element in the sequence is copied to the output as is.
In the ``exclusive`` mode the ``j-th`` output element is the sum of the first ``j-1`` elements and the first element in the output sequence is ``0``.
To perform the summation in the opposite direction of the axis, set reverse attribute to ``true``.

**Attributes**:

* *exclusive*

  * **Description**: If the attribute is set to ``true``, then exclusive sums are returned, the ``j-th`` element is not included in the ``j-th`` sum. Otherwise, the inclusive sum of the first ``j`` elements for the ``j-th`` element is calculated.
  * **Range of values**:

    * ``false`` - include the top element
    * ``true`` - do not include the top element
  * **Type**: ``boolean``
  * **Default value**: ``false``
  * **Required**: *no*

* *reverse*

  * **Description**: If set to ``true`` will perform the sums in reverse direction.
  * **Range of values**:

    * ``false`` - do not perform sums in reverse direction
    * ``true`` - perform sums in reverse direction
  * **Type**: ``boolean``
  * **Default value**: ``false``
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* and rank greater or equal to 1. **Required.**
* **2**: Axis index along which the cumulative sum is performed. A scalar of type *T_AXIS*. Negative value means counting dimensions from the back. Default value is ``0``. **Optional.**

**Outputs**

* **1**: Output tensor with cumulative sums of the input elements. A tensor of type *T* of the same shape as the first input.

**Types**

* *T*: any numeric type.

* *T_AXIS*: ``int64`` or ``int32``.

**Examples**

*Example 1*

.. code-block:: xml
   :force:

   <layer ... type="CumSum" exclusive="0" reverse="0">
       <input>
           <port id="0">     <!-- input value is: [1., 2., 3., 4., 5.] -->
               <dim>5</dim>
           </port>
           <port id="1"/>     <!-- axis value is: 0 -->
       </input>
       <output>
           <port id="2">     <!-- output value is: [1., 3., 6., 10., 15.] -->
               <dim>5</dim>
           </port>
       </output>
   </layer>

*Example 2*

.. code-block:: xml
   :force:

   <layer ... type="CumSum" exclusive="1" reverse="0">
       <input>
           <port id="0">     <!-- input value is: [1., 2., 3., 4., 5.] -->
               <dim>5</dim>
           </port>
           <port id="1"/>     <!-- axis value is: 0 -->
       </input>
       <output>
           <port id="2">     <!-- output value is: [0., 1., 3., 6., 10.] -->
               <dim>5</dim>
           </port>
       </output>
   </layer>

*Example 3*

.. code-block:: xml
   :force:

   <layer ... type="CumSum" exclusive="0" reverse="1">
       <input>
           <port id="0">     <!-- input value is: [1., 2., 3., 4., 5.] -->
               <dim>5</dim>
           </port>
           <port id="1"/>     <!-- axis value is: 0 -->
       </input>
       <output>
           <port id="2">     <!-- output value is: [15., 14., 12., 9., 5.] -->
               <dim>5</dim>
           </port>
       </output>
   </layer>

*Example 4*

.. code-block:: xml
   :force:

   <layer ... type="CumSum" exclusive="1" reverse="1">
       <input>
           <port id="0">     < -- input value is: [1., 2., 3., 4., 5.] -->
               <dim>5</dim>
           </port>
           <port id="1"/>     < -- axis value is: 0 -->
       </input>
       <output>
           <port id="2">     < -- output value is: [14., 12., 9., 5., 0.] -->
               <dim>5</dim>
           </port>
       </output>
   </layer>


