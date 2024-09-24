Range
=====


.. meta::
  :description: Learn about Range-1 - a generation operation, which can be
                performed on three required input tensors.

**Versioned name**: *Range-1*

**Category**: *Generation*

**Short description**: *Range* operation generates a sequence of numbers according input values [start, stop) with a step.

**Attributes**:

No attributes available.

**Inputs**:

* **1**: "start" - A scalar of type *T*. **Required.**
* **2**: "stop" - A scalar of type *T*. **Required.**
* **3**: "step" - A scalar of type *T*. **Required.**

**Outputs**:

* **1**: A tensor of type *T*.

**Types**

* *T*: any numeric type.

**Detailed description**:

*Range* operation generates a sequence of numbers starting from the value in the first input (start) up to but not including the value in the second input (stop) with a step equal to the value in the third input, according to the following formula:

For a positive ``step``:

.. math::

   start<=val[i]<stop,


for a negative ``step``:

.. math::

   start>=val[i]>stop,


where

.. math::

   val[i]=start+i*step


**Examples**

*Example 1: positive step*

.. code-block:: xml
   :force:

    <layer ... type="Range">
        <input>
            <port id="0">  <!-- start value: 2 -->
            </port>
            <port id="1">  <!-- stop value: 23 -->
            </port>
            <port id="2">  <!-- step value: 3 -->
            </port>
        </input>
        <output>
            <port id="3">
                <dim>7</dim> <!-- [ 2,  5,  8, 11, 14, 17, 20] -->
            </port>
        </output>
    </layer>


*Example 2: negative step*

.. code-block:: xml
   :force:

    <layer ... type="Range">
        <input>
            <port id="0">  <!-- start value: 23 -->
            </port>
            <port id="1">  <!-- stop value: 2 -->
            </port>
            <port id="2">  <!-- step value: -3 -->
            </port>
        </input>
        <output>
            <port id="3">
                <dim>7</dim> <!-- [23, 20, 17, 14, 11, 8, 5] -->
            </port>
        </output>
    </layer>

