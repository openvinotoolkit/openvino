FloorMod
========


.. meta::
  :description: Learn about FloorMod-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *FloorMod-1*

**Category**: *Arithmetic binary*

**Short description**: *FloorMod* performs an element-wise floor modulo operation with two given tensors applying broadcasting rule specified in the *auto_broadcast* attribute.

**Detailed description**:
As a first step input tensors *a* and *b* are broadcasted if their shapes differ. Broadcasting is performed according to `auto_broadcast` attribute specification. As a second step *FloorMod* operation is computed element-wise on the input tensors *a* and *b* according to the formula below:

.. math::

   o_{i} = a_{i} \mod b_{i}


*FloorMod* operation computes a reminder of a floored division. It is the same behavior like in
Python programming language: :math:`floor(x / y) * y + floor\_mod(x, y) = x`. The sign of the result is equal to a sign of a divisor. The result of division by zero is undefined.

**Attributes**:

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes must match
    * *numpy* - numpy broadcasting rules, description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`

  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**
* **2**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise floor modulo operation. A tensor of type *T* with shape equal to broadcasted shape of two inputs.

**Types**

* *T*: any numeric type.

**Examples**

*Example 1 - no broadcasting*

.. code-block:: xml
   :force:

   <layer ... type="FloorMod">
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

   <layer ... type="FloorMod">
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


