Range
=====


.. meta::
  :description: Learn about Range-4 - a generation operation, which can be
                performed on three required input tensors.

**Versioned name**: *Range-4*

**Category**: *Generation*

**Short description**: *Range* operation generates a sequence of numbers according input values [start, stop) with a step.

**Attributes**:

* *output_type*

  * **Description**: the output tensor type
  * **Range of values**: any numeric type
  * **Type**: string
  * **Required**: *yes*

**Inputs**:

* **1**: "start" - A scalar of type *T1*. **Required.**
* **2**: "stop" - A scalar of type *T2*. **Required.**
* **3**: "step" - A scalar of type *T3*. If ``step`` is equal to zero after casting to ``output_type``, behavior is undefined. **Required.**

**Outputs**:

* **1**: A tensor with type specified by attribute *output_type*.

**Types**

* *T1*, *T2*, *T3*: any numeric type.

**Detailed description**:

*Range* operation generates a sequence of numbers starting from the value in the first input (``start``) up to but not including the value in the second input (``stop``) with a ``step`` equal to the value in the third input, according to the following formula:

For a positive ``step``:

.. math::

   start<=val[i]<stop,


for a negative ``step``:

.. math::

   start>=val[i]>stop,


the i-th element is calculated by the following formula:

.. math::

   val[i+1]=val[i]+step.


The calculations are done after casting all values to ``accumulate_type(output_type)``. ``accumulate_type`` is a type that have better or equal accuracy for accumulation than ``output_type`` on current hardware, e.g. ``fp64`` for ``fp16``. The number of elements is calculated in the floating-point type according to the following formula:

.. math::

   max(ceil((end − start) / step), 0)


This is aligned with PyTorch's operation ``torch.arange``, to align with tensorflow operation ``tf.range`` all inputs must be casted to ``output_type`` before calling *Range*. The rounding for casting values are done towards zero.

**Examples**

*Example 1: positive step*

.. code-block:: xml
   :force:

    <layer ... type="Range">
        <data output_type="i32">
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
        <data output_type="i32">
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


*Example 3: floating-point*

.. code-block:: xml
   :force:

    <layer ... type="Range">
        <data output_type="f32">
        <input>
            <port id="0">  <!-- start value: 1 -->
            </port>
            <port id="1">  <!-- stop value: 2.5 -->
            </port>
            <port id="2">  <!-- step value: 0.5 -->
            </port>
        </input>
        <output>
            <port id="3">
                <dim>3</dim> <!-- [ 1.0,  1.5,  2.0] -->
            </port>
        </output>
    </layer>

