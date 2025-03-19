Power
=====


.. meta::
  :description: Learn about Power-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Power-1*

**Category**: *Arithmetic binary*

**Short description**: *Power* performs element-wise power operation with two given tensors applying broadcasting rule specified in the *auto_broadcast* attribute.

**Detailed description**
As a first step input tensors *a* and *b* are broadcasted if their shapes differ. Broadcasting is performed according to `auto_broadcast` attribute specification. As a second step *Power* operation is computed element-wise on the input tensors *a* and *b* according to the formula below:

.. math::

    o_i = a_i^{b_i}

**Attributes**:

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes must match
    * *numpy* - numpy broadcasting rules, description is available in   description is available in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`
  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**
* **2**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise power operation. A tensor of type *T* with shape equal to broadcasted shape of two inputs.

**Types**

* *T*: any numeric type.


**Examples**

*Example 1 - no broadcasting*

.. code-block:: xml
   :force:

    <layer ... type="Power">
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

    <layer ... type="Power">
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



