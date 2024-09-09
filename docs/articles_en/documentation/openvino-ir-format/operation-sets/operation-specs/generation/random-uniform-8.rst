RandomUniform
=============


.. meta::
  :description: Learn about RandomUniform-8 - a generation operation, which can be
                performed on three required input tensors.

**Versioned name**: *RandomUniform-8*

**Category**: *Generation*

**Short description**: *RandomUniform* operation generates a sequence of random values from a uniform distribution.

**Detailed description**:

*RandomUniform* operation generates random numbers from a uniform distribution in the range ``[minval, maxval)``.
The generation algorithm is based on underlying random integer generator that uses Philox algorithm. Philox algorithm
is a counter-based pseudo-random generator, which produces uint32 values. Single invocation of Philox algorithm returns
four result random values, depending on the given *key* and *counter* values. *Key* and *counter* are initialized
with *global_seed* and *op_seed* attributes respectively.

If both seed values equal to zero, RandomUniform generates non-deterministic sequence.

.. math::

   key = global_seed\\
   counter = op_seed


Link to the original paper `Parallel Random Numbers: As Easy as 1, 2, 3 <https://www.thesalmons.org/john/random123/papers/random123sc11.pdf>`__.

The result of Philox is calculated by applying a fixed number of *key* and *counter* updating so-called "rounds".
This implementation uses 4x32_10 version of Philox algorithm, where number of rounds = 10.

Suppose we have *n* which determines *n*-th 4 elements of random sequence.
In each round *key*, *counter* and *n* are splitted to pairs of uint32 values:

.. math::

   R = cast\_to\_uint32(value)\\
   L = cast\_to\_uint32(value >> 32),

where *cast\_to\_uint32* - static cast to uint32, *value* - uint64 input value, *L*, *R* - uint32
result values, >> - bitwise right shift.

Then *n* and *counter* are updated with the following formula:

.. math::

   L'= mullo(R, M)\\
   R' = mulhi(R, M) {\oplus} k {\oplus} L \\
   mulhi(a, b) = floor((a {\times} b) / 2^{32}) \\
   mullo(a, b) = (a {\times} b) \mod 2^{32}

where :math:`{\oplus}` - bitwise xor, *k* = :math:`R_{key}` for updating counter, *k* = :math:`L_{key}` for updating *n*, *M* = ``0xD2511F53`` for updating *n*, *M* = ``0xCD9E8D57`` for updating *counter*.

After each round *key* is raised by summing with another pair of const values:

.. math::

   L += 0x9E3779B9 \\
   R += 0xBB67AE85

Values :math:`L'_{n}, R'_{n}, L'_{counter}, R'_{counter}` are resulting four random numbers.

Float values between [0..1) are obtained from 32-bit integers by the following rules.

Float16 is formatted as follows: *sign* (1 bit) *exponent* (5 bits) *mantissa* (10 bits). The value is interpreted
using following formula:

.. math::

   (-1)^{sign} * 1, mantissa * 2 ^{exponent - 15}


so to obtain float16 values *sign*, *exponent* and *mantissa* are set as follows:

.. code-block:: xml
   :force:

   sign = 0
   exponent = 15 - representation of a zero exponent.
   mantissa = 10 right bits from generated uint32 random value.


So the resulting float16 value is:

.. code-block:: xml
   :force:

   x_uint16 = x // Truncate the upper 16 bits.
   val = ((exponent << 10) | x_uint16 & 0x3ffu) - 1.0,

where x is uint32 generated random value.

Float32 is formatted as follows: *sign* (1 bit) *exponent* (8 bits) *mantissa* (23 bits). The value is interpreted using following formula:

.. math::

   (-1)^{sign} * 1, mantissa * 2 ^{exponent - 127}


so to obtain float values *sign*, *exponent* and *mantissa* are set as follows:

.. code-block:: xml
   :force:

   sign = 0
   exponent = 127 - representation of a zero exponent.
   mantissa = 23 right bits from generated uint32 random value.


So the resulting float value is:

.. code-block:: xml
   :force:

   val = ((exponent << 23) | x & 0x7fffffu) - 1.0,

where x is uint32 generated random value.

Double is formatted as follows: *sign* (1 bit) *exponent* (11 bits) *mantissa* (52 bits). The value is interpreted using following formula:

.. math::

   (-1)^{sign} * 1, mantissa * 2 ^{exponent - 1023}


so to obtain double values *sign*, *exponent* and *mantissa* are set as follows:

.. code-block:: xml
   :force:

   sign = 0
   exponent = 1023 - representation of a zero exponent.
   mantissa = 52 right bits from two concatinated uint32 values from random integer generator.


So the resulting double is obtained as follows:

.. code-block:: xml
   :force:

   mantissa_h = x0 & 0xfffffu;  // upper 20 bits of mantissa
   mantissa_l = x1;             // lower 32 bits of mantissa
   mantissa = (mantissa_h << 32) | mantissa_l;
   val = ((exponent << 52) | mantissa) - 1.0,

where x0, x1 are uint32 generated random values.

To obtain a value in a specified range each value is processed with the following formulas:

For float values:

.. math::

   result = x * (maxval - minval) + minval,

where *x* is random float or double value between [0..1).

For integer values:

.. math::

   result = x \mod (maxval - minval) + minval,

where *x* is uint32 random value.


Example 1. *RandomUniform* output with ``global_seed`` = 150, ``op_seed`` = 10, ``output_type`` = f32:

.. code-block:: xml
   :force:

    input_shape    = [ 3, 3 ]
    output  = [[0.7011236  0.30539632 0.93931055]
            [0.9456035   0.11694777 0.50770056]
            [0.5197197   0.22727466 0.991374  ]]


Example 2. *RandomUniform* output with ``global_seed`` = 80, ``op_seed`` = 100, ``output_type`` = double:

.. code-block:: xml
   :force:

   input_shape    = [ 2, 2 ]

   minval = 2

   maxval = 10

   output  = [[5.65927959 4.23122376]
         [2.67008206 2.36423758]]


Example 3. *RandomUniform* output with ``global_seed`` = 80, ``op_seed`` = 100, ``output_type`` = i32:

.. code-block:: xml
   :force:

   input_shape    = [ 2, 3 ]

   minval = 50

   maxval = 100

   output  = [[65 70 56]
         [59 82 92]]


**Attributes**:

* ``output_type``

  * **Description**: the type of the output. Determines generation algorithm and affects resulting values. Output numbers generated for different values of *output_type* may not be equal.
  * **Range of values**: "i32", "i64", "f16", "bf16", "f32", "f64".
  * **Type**: string
  * **Required**: *Yes*

* ``global_seed``

  * **Description**: global seed value.
  * **Range of values**: positive integers
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *Yes*

* ``op_seed``

  * **Description**: operational seed value.
  * **Range of values**: positive integers
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *Yes*

**Inputs**:

*   **1**: ``shape`` - 1D tensor of type *T_SHAPE* describing output shape. **Required.**

*   **2**: ``minval`` - scalar or 1D tensor with 1 element with type specified by the attribute *output_type*, defines the lower bound on the range of random values to generate (inclusive). **Required.**

*   **3**: ``maxval`` - scalar or 1D tensor with 1 element with type specified by the attribute *output_type*, defines the upper bound on the range of random values to generate (exclusive). **Required.**


**Outputs**:

* **1**: A tensor with type specified by the attribute *output_type* and shape defined by ``shape`` input tensor.

**Types**

* *T_SHAPE*: ``int32`` or ``int64``.

*Example 1: IR example.*

.. code-block:: xml
   :force:

    <layer ... name="RandomUniform" type="RandomUniform">
        <data output_type="f32" global_seed="234" op_seed="148"/>
        <input>
            <port id="0" precision="I32">  <!-- shape value: [2, 3, 10] -->
                <dim>3</dim>
            </port>
            <port id="1" precision="FP32"/> <!-- min value -->
            <port id="2" precision="FP32"/> <!-- max value -->
        </input>
        <output>
            <port id="3" precision="FP32" names="RandomUniform:0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>10</dim>
            </port>
        </output>
    </layer>



