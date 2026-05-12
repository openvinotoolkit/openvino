Atan2
=====


.. meta::
  :description: Learn about Atan2-17 - an element-wise, arithmetic operation, which
                can be performed on two tensors in OpenVINO.

**Versioned name**: *Atan2-17*

**Category**: *Arithmetic binary*

**Short description**: *Atan2* performs element-wise four-quadrant arctangent operation on two input tensors applying broadcasting rule specified in the *auto_broadcast* attribute.

**Detailed description**:
As a first step input tensors *y* and *x* are broadcasted if their shapes differ. Broadcasting is performed according to ``auto_broadcast`` attribute specification. As a second step *Atan2* operation is computed element-wise on the input tensors *y* and *x* according to the formula below:

.. math::

   o_{i} = \text{atan2}(y_{i},\ x_{i})

The result is the angle in radians between the positive x-axis and the point :math:`(x_i, y_i)`, in the range :math:`[-\pi, \pi]`.

Only floating-point input types are supported.

**Attributes**:

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes must match
    * *numpy* - numpy broadcasting rules, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`
    * *pdpd* - PaddlePaddle-style broadcasting rules, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`

  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape representing the y (ordinate) values. **Required.**
* **2**: A tensor of type *T* and arbitrary shape representing the x (abscissa) values. **Required.**

**Outputs**

* **1**: The result of element-wise Atan2 operation. A tensor of type *T* with shape equal to broadcasted shape of the two inputs.

**Types**

* *T*: any supported floating-point type.

**Examples**

*Example 1 - no broadcasting*

.. code-block:: xml
   :force:

   <layer ... type="Atan2" version="opset17">
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

*Example 2: numpy broadcasting*

.. code-block:: xml
   :force:

   <layer ... type="Atan2" version="opset17">
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
