RandomUniform
=============


.. meta::
  :description: Learn about RandomUniform-8 - a generation operation, which can be
                performed on three required input tensors.

**Versioned name**: *RandomUniform-8*

**Category**: *Generation*

**Short description**: *RandomUniform* operation generates a sequence of random values
from a uniform distribution.

**Detailed description**:

*RandomUniform* operation generates random numbers from a uniform distribution in the range
``[minval, maxval)``. The generation algorithm is based on an underlying random integer
generator that uses either Philox or Mersenne-Twister algorithm. Both algorithms are
counter-based pseudo-random generators, which produce uint32 values.
A single algorithm invocation returns four result random values, depending on the
given initial values. For Philox, these values are *key* and *counter*,
for Mersenne-Twister it is a single *state* value. *Key* and *counter* are initialized
with *global_seed* and *op_seed* attributes respectively, while the *state* is only
initialized using *global_seed*.

Algorithm selection allows aligning the output of OpenVINO's Random Uniform op with
the ones available in Tensorflow and PyTorch.
The *alignment* attribute selects which framework the output should be aligned to.
Tensorflow uses the Philox algorithm and PyTorch uses the Mersenne-Twister algorithm.
For Tensorflow, this function is equivalent to the function
``tf.raw_ops.RandomUniform(shape, dtype, global_seed, op_seed)`` when dtype represents a real
number, and ``tf.raw_ops.RandomUniformInt(shape, min_val, max_val, dtype, global_seed,
op_seed)`` for integer types. Internally, both of these functions are executed by
``tf.random.uniform(shape, min_val, max_val, dtype, global_seed, op_seed)``, where for
floating-point dtype the output goes through additional conversion to reside within a given range.

For PyTorch, this function is equivalent to the
``torch.Tensor(shape, dtype).uniform_(min_val, max_val)`` function when dtype represents
a real number, and ``torch.Tensor(shape, dtype).random_(min_val, max_val)`` for integer
types. Internally, both of these functions are executed by ``torch.rand(shape, dtype)``
with default generator and layout. The seed of these functions is provided by calling
``torch.manual_seed(global_seed)``. The ``op_seed`` value is ignored.
By default, the output is aligned with Tensorflow (Philox algorithm). T
his behavior is backwards-compatible.

If both seed values are equal to zero, RandomUniform generates a non-deterministic sequence.

**Philox Algorithm Explanation**:

.. math::

   key = global\_seed\\
   counter = op\_seed


Link to the original paper
`Parallel Random Numbers: As Easy as 1, 2, 3 <https://www.thesalmons.org/john/random123/papers/random123sc11.pdf>`__.

The result of Philox is calculated by applying a fixed number of *key* and *counter* updating so-called "rounds".
This implementation uses 4x32_10 version of Philox algorithm, where number of rounds = 10.

Suppose we have *n* which determines *n*-th 4 elements of random sequence.
In each round, *key*, *counter* and *n* are split to pairs of uint32 values:

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


where :math:`{\oplus}` - bitwise xor, *k* = :math:`R_{key}` for updating counter,
*k* = :math:`L_{key}` for updating *n*, *M* = ``0xD2511F53`` for updating *n*,
*M* = ``0xCD9E8D57`` for updating *counter*.

After each round *key* is raised by summing with another pair of const values:

.. math::

   L += 0x9E3779B9 \\
   R += 0xBB67AE85


Values :math:`L'_{n}, R'_{n}, L'_{counter}, R'_{counter}` are resulting four random numbers.

Float values between [0..1) are obtained from 32-bit integers by the following rules.

Float16 is formatted as follows: *sign* (1 bit) *exponent* (5 bits) *mantissa* (10 bits).
The value is interpreted using following formula:

.. math::

   (-1)^{sign} * 1, mantissa * 2 ^{exponent - 15}


so to obtain float16 values *sign*, *exponent* and *mantissa* are set as follows:

.. code-block:: cpp
   :force:

   sign = 0
   exponent = 15 // representation of a zero exponent.
   mantissa = 10 // right bits from generated uint32 random value.


So the resulting float16 value is:

.. code-block:: cpp
   :force:

   x_uint16 = x // Truncate the upper 16 bits.
   val = ((exponent << 10) | x_uint16 & 0x3ffu) - 1.0,


where ``x`` is uint32 generated random value.

Float32 is formatted as follows: *sign* (1 bit) *exponent* (8 bits) *mantissa* (23 bits).
The value is interpreted using following formula:

.. math::

   (-1)^{sign} * 1, mantissa * 2 ^{exponent - 127}


so to obtain float values *sign*, *exponent* and *mantissa* are set as follows:

.. code-block:: xml
   :force:

   sign = 0
   exponent = 127 - representation of a zero exponent.
   mantissa = 23 right bits from generated uint32 random value.


So the resulting float value is:

.. code-block:: cpp
   :force:

   val = ((exponent << 23) | x & 0x7fffffu) - 1.0,


where ``x`` is uint32 generated random value.

Double is formatted as follows: *sign* (1 bit) *exponent* (11 bits) *mantissa* (52 bits).
The value is interpreted using following formula:

.. math::

   (-1)^{sign} * 1, mantissa * 2 ^{exponent - 1023}


so to obtain double values *sign*, *exponent* and *mantissa* are set as follows:

.. code-block:: cpp
   :force:

   sign = 0
   exponent = 1023 // representation of a zero exponent.
   mantissa = 52 // right bits from two concatenated uint32 values from random integer generator.


So the resulting double is obtained as follows:

.. code-block:: cpp
   :force:

   mantissa_h = x0 & 0xfffffu; // upper 20 bits of mantissa
   mantissa_l = x1; // lower 32 bits of mantissa
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


**Example 1.** *RandomUniform* output with ``global_seed = 150``, ``op_seed = 10``,
``output_type = f32``, ``alignment = TENSORFLOW``:

.. code-block:: cpp
   :force:

    input_shape = [3, 3]
    output  = [[0.7011236  0.30539632 0.93931055]\
            [0.9456035   0.11694777 0.50770056]\
            [0.5197197   0.22727466 0.991374  ]]


**Example 2.** *RandomUniform* output with ``global_seed = 80``, ``op_seed = 100``,
``output_type = double``, ``alignment = TENSORFLOW``:

.. code-block:: cpp
   :force:

   input_shape  = [2, 2]
   minval = 2
   maxval = 10
   output  = [[5.65927959 4.23122376]\
         [2.67008206 2.36423758]]


**Example 3**. *RandomUniform* output with ``global_seed = 80``,
``op_seed = 100``, ``output_type = i32``, ``alignment = TENSORFLOW``:

.. code-block:: cpp
   :force:

   input_shape = [ 2, 3 ]
   minval = 50
   maxval = 100
   output  = [[65 70 56]\
         [59 82 92]]


Mersenne-Twister Algorithm Explanation:
#######################################

| Link to the original paper Mersenne Twister:
| `Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator <https://dl.acm.org/doi/10.1145/272991.272995>`__.

The Mersenne-Twister algorithm generates random numbers by initializing a state array
with a seed and then iterating through a series of transformations.
Suppose we have n which determines the n-th element of the random sequence.

The initial state array is generated recursively using the following formula:

.. code-block:: cpp
   :force:

   state[0] = global_seed & 0xffffffff;
   state[i] = 1812433253 * state[i-1] ^ (state[i-1] >> 30) + i


where the value of i cannot exceed 623.

The output is generated by tempering the state array:

.. math::

   \begin{align}
   &y = state[i]\\
   &y = y \oplus (y >> u)\\
   &y = y \oplus ((y << s) \& b)\\
   &y = y \oplus ((y << t) \& c)\\
   &y = y \oplus (y >> l)
   \end{align}


where ``u``, ``s``, ``t``, ``l``, ``b``, and ``c`` are constants.

Whenever all state values are 'used', a new state array is generated recursively as follows:

.. code-block:: cpp
   :force:

   current_state = state[i]
   next_state    = state[i+1] if i+1 <= 623 else state[0]
   next_m_state  = state[i+m] if i+m <= 623 else state[i+m-623]
   twisted_state = (((current_state & 0x80000000) | (next_state & 0x7fffffff)) >> 1) ^ (next_state & 1 ? 0x9908b0df : 0)
   state[i] = next_m_state ^ twisted_state


where ``m`` is a constant.

For parity with PyTorch, the value of the constants is set as follows:

.. code-block:: cpp
   :force:

   u = 11
   s = 7
   b = 0x9d2c5680
   t = 15
   c = 0xefc60000
   l = 18
   m = 397


These values follow the recommendations from the linked paper for MT19937.

To convert a given unsigned int value (denoted as ``x`` below) to a specific type,
a simple conversion is performed.

For float32:

.. code-block:: cpp
   :force:

   mantissa_digits = 24 //(mantissa / significand bits count of float + 1, equal to std::numeric_limits<float>::digits == FLT_MANT_DIG == 24)
   mask = uint32(uint64(1) << mantissa_digits - 1)
   divisor = float(1) / (uint64(1) << mantissa_digits)
   output = float((x & mask) * divisor)


For float16:

.. code-block:: cpp

   mantissa_digits = 11 //(mantissa / significand bits count of float16 + 1, equal to 11)
   mask = uint32(uint64(1) << mantissa_digits - 1)
   divisor = float(1) / (uint64(1) << mantissa_digits)
   output = float16((x & mask) * divisor)


For bfloat16:

.. code-block:: cpp
   :force:

   mantissa_digits = 8 //(mantissa / significand bits count of bfloat16 + 1, equal to 8)
   mask = uint32(uint64(1) << mantissa_digits - 1)
   divisor = float(1) / (uint64(1) << mantissa_digits)
   output = bfloat16((x & mask) * divisor)


For float64 (double precision requires the use of two uint32 values, denoted as
x and y below):

.. code-block:: cpp
   :force:

   value = uint64(x) << 32 + y

   mantissa_digits = 53 //(mantissa / significand bits count of double + 1, equal to std::numeric_limits<double>::digits == DBL_MANT_DIG == 53)
   mask = uint64(1) << mantissa_digits - 1
   divisor = double(1) / (uint64(1) << mantissa_digits)
   output = double((x & mask) * divisor)


All of the floating - point types above after the conversion fall between the values
of 0 and 1. To convert them to reside between a range *<min, max>*, a simple operation
is performed:

.. math::

   output = x * (max - min) + min


For integer types, no special conversion operation is done except for int64 when
either min or max exceeds the maximum possible value of uint32. A simple operation to
standardize the values is performed.
The special behavior (optimization) for int64 matches the expected output for PyTorch,
normally a concatenation of 2 uint32s always occurs.
In other words:

.. code-block:: cpp
   :force:

   if output is of int32 dtype:
      output = int32(x)
   else if output is of int64 dtype and (min <= max(uint32) and max <= max(uint32)):
      output = int64(x)
   else:
      output = int64(x << 32 + y) (uses 2 uint32s instead of one)
   output = output % (max - min) + min


**Example 1.** RandomUniform output with ``initial_seed = 150``, ``output_type = f32``,
``alignment = PYTORCH``:

.. code-block:: cpp
   :force:

   input_shape  = [ 3, 3 ]
   output  = [[0.6789123  0.31274895 0.91842768] \
              [0.9312087   0.13456984 0.49623574] \
              [0.5082716   0.23938411 0.97856429]]


**Example 2.** RandomUniform output with ``initial_seed = 80``, ``output_type = double``,
``alignment = PYTORCH``:

.. code-block:: cpp
   :force:

   input_shape = [ 2, 2 ]
   minval = 2
   maxval = 10
   output  = [[8.34928537 6.12348725] \
              [3.76852914 2.89564172]]


**Example 3.** RandomUniform output with ``initial_seed = 80``, ``output_type = i32``,
``alignment = PYTORCH``:

.. code-block:: cpp
   :force:

   input_shape = [ 2, 3 ]
   minval = 50
   maxval = 100
   output  = [[89 73 68] \
              [95 78 61]]


**Attributes**:

* ``output_type``

  * **Description**: the type of the output. Determines generation algorithm and affects
    resulting values. Output numbers generated for different values of *output_type* may
    not be equal.
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

* ``alignment``

  * **Description**: the framework to align the output to.
  * **Range of values**: TENSORFLOW, PYTORCH
  * **Type**: `string`
  * **Default value**: TENSORFLOW
  * **Required**: *No*

**Inputs**:

* **1**: ``shape`` - 1D tensor of type *T_SHAPE* describing output shape. **Required.**

* **2**: ``minval`` - scalar or 1D tensor with 1 element with type specified by the
  attribute *output_type*, defines the lower bound on the range of random values to
  generate (inclusive). **Required.**

* **3**: ``maxval`` - scalar or 1D tensor with 1 element with type specified by the
  attribute *output_type*, defines the upper bound on the range of random values to
  generate (exclusive). **Required.**


**Outputs**:

* **1**: A tensor with type specified by the attribute *output_type* and shape defined
  by ``shape`` input tensor, with values aligned to the framework selected by the
  ``alignment`` attribute.

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
