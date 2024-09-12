Multinomial
===========


.. meta::
  :description: Learn about Multinomial-13 - a generation operation, that creates a sequence of indices of classes sampled from the multinomial distribution.

**Versioned name**: *Multinomial-13*

**Category**: *Generation*

**Short description**: *Multinomial* operation generates a sequence of class indices sampled from the multinomial distribution based on the input class probabilities.

**Detailed description**: *Multinomial* operation generates a sequence of class indices sampled from the multinomial distribution. In this context, the *probs* values represent the probabilities associated with each class within the multinomial distribution. When executing this operation, it randomly selects a class based on these probabilities. Subsequently, the index of the chosen class in the *probs* array is appended to the *output* sequence in the corresponding batch.

**Algorithm formulation**:

.. note::

  The following notation denotes a range of real numbers between a and b.

.. math::

   [a, b] => { x \in \mathbb{R},  a <= x <= b }


Given a list of probabilities x1, x2, ..., xn:

* If *log_probs* is true:

  * For each probability x, replace it with a value :math:`e^{x}`.

* Create an array - discrete CDF (`Cumulative Distribution Function <https://hal.science/hal-00753950/file/PEER_stage2_10.1016%252Fj.spl.2011.03.014.pdf>`__) - the cumulative sum of those probabilities, ie. create an array of values where the ith value is the sum of the probabilities x1, ..., xi.
* Divide the created array by its maximum value to normalize the cumulative probabilities between the real values in the range [0, 1]. This array is, by definition of CDF, sorted in ascending order, hence the maximum value is the last value of the array.
* Randomly generate a sequence of double-precision floating point numbers in the range [0, 1].
* For each generated number, assign the class with the lowest index for which the cumulative probability is less or equal to the generated value.
* If *with_replacement* is False (sampling without replacement):

  * Assume a class with index i has been selected - then every CDF value starting at i-th index should be lowered by the original probability of the selected class. This effectively sets the probability of sampling the given class to 0.
  * Afterwards, divide the CDF by its last (maximum) value to normalize the cumulative probabilities between the real values in the range [0, 1].

* Convert the output indices to *convert_type*.
* Return output indices.

**Example computations**:

Example 1 - simple 2D tensor with one batch

* Let ``probs`` = ``[[0.1, 0.5, 0.4]]``, ``num_samples`` = 5, ``log_probs`` = false, ``with_replacement`` = true
* CDF of ``probs`` = ``[[0.1, 0.1 + 0.5, 0.1 + 0.5 + 0.4]]`` = ``[[0.1, 0.6, 1]]``
* Randomly generated floats = ``[[0.2, 0.4, 0.6, 0.8, 1]]``
* Assigned classes = ``[[1, 1, 1, 2, 2]]``

Example 2 - 2D tensor, log probabilities

* Let ``probs`` = ``[[-1, 1, 2], [50, 1, 21]]``, ``num_samples`` = 10, ``log_probs`` = true, ``with_replacement`` = true
* Exponentiated ``probs`` = ``[[0.36, 2.71, 7.38], [5184705528587072464087.45, 2.71, 1318815734.48]]``
* CDF of ``probs``, per batch = ``[[0.36, 3.07, 10.45], [5184705528587072464087.45, 5184705528587072464090.16, 5184705528588391279824.64]]``
* Normalized CDF = ``[[0.03, 0.29, 1], [1.0, 1.0, 1.0]]``
* Randomly generated floats = ``[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]``
* Assigned classes = ``[[1, 1, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]``

Example 3 - 2D tensor, without replacement

* Let ``probs`` = ``[[0.1, 0.5, 0.4]]``, ``num_samples`` = 2, ``log_probs`` = false, ``with_replacement`` = false
* CDF of ``probs`` = ``[[0.1, 0.6, 1]]``
* Randomly generated floats = ``[[0.3, 0.2]]``
* In a loop:

  * For a value of 0.3, a class with idx ``1`` is selected
  * Therefore, in CDF, for every class starting with idx ``1`` subtract the probability of class at idx ``1`` = ``probs[1]`` = 0.5
  * CDF = ``[[0.1, 0.6 - 0.5, 1.0 - 0.5]]`` = ``[[0.1, 0.1, 0.5]]``
  * Normalize CDF by dividing by last value: CDF = ``[[0.2, 0.2, 1.0]]``
  * Take the next randomly generated float, here 0.2, and repeat until all random samples have assigned classes. Notice that for ``sampled values`` <= 0.2, only the class with idx ``0`` can be selected, since the search stops at the index with the first value satisfying ``sample value`` <= ``CDF probability``

* Assigned classes = ``[[1, 2]]``


**Attributes**:

* ``convert_type``

  * **Description**: the type of the output. Determines generation algorithm and affects resulting values. Output numbers generated for different values of *convert_type* may not be equal.
  * **Range of values**: "i32", "i64".
  * **Type**: string
  * **Required**: *Yes*

* ``with_replacement``

  * **Description**: controls whether to sample with replacement (classes can be sampled multiple times).
  * **Range of values**: `true`, `false`

    * ``true`` - class indices can be sampled multiple times.
    * ``false`` - class indices will not repeat in the output and the size of ``probs``' ``class_size`` dimension is required to be larger or equal to *num_samples* value. Might affect performance.

  * **Type**: `bool`
  * **Required**: *Yes*

* ``log_probs``

  * **Description**: allows to control whether *inputs* should be treated as probabilities or unnormalized log probabilities.
  * **Range of values**: `true`, `false`

    * ``true`` - set values in *inputs* are unnormalized log probabilities that can be any real number.
    * ``false`` - probabilities in *inputs* are expected to be non-negative, finite and have a non-zero-sum.

  * **Type**: `bool`
  * **Required**: *Yes*

* ``global_seed``

  * **Description**: global seed value.
  * **Range of values**: non-negative integers
  * **Type**: `unsigned int 64-bit`
  * **Default value**: 0
  * **Required**: *No*

* ``op_seed``

  * **Description**: operational seed value.
  * **Range of values**: non-negative integers
  * **Type**: `unsigned int 64-bit`
  * **Default value**: 0
  * **Required**: *No*

**Inputs**:

*   **1**: ``probs`` - A 2D tensor of type `T_IN` and shape `[batch_size, class_size]` with probabilities. Allowed values depend on the *log_probs* attribute. The values are internally normalized to have values in the range of `[0, 1]` with the sum of all probabilities in the given batch equal to 1. **Required.**

*   **2**: ``num_samples`` - A scalar or 1D tensor with a single element of type `T_SAMPLES` specifying the number of samples to draw from the multinomial distribution. **Required.**

**Outputs**:

* **1**:  ``output``-  A tensor with type specified by the attribute *convert_type* and shape ``[batch_size, num_samples]``.

**Types**

* **T_IN**: any supported floating-point type.
* **T_SAMPLES**: 32-bit or 64-bit integers.


*Example 1: 2D input tensor with one batch.*

.. code-block:: xml
   :force:

    <layer ... name="Multinomial" type="Multinomial">
        <data convert_type="f32", with_replacement="true", log_probs="false", global_seed="234", op_seed="148"/>
        <input>
            <port id="0" precision="FP32">  <!-- probs value: [[0.1, 0.5, 0.4]] -->
                <dim>1</dim> <!-- batch size of 2 -->
                <dim>3</dim>
            </port>
            <port id="1" precision="I32"/> <!-- num_samples value: 5 -->
        </input>
        <output>
            <port id="3" precision="I32" names="Multinomial:0">
                <dim>1</dim> <!--dimension depends on input batch size -->
                <dim>5</dim> <!--dimension depends on num_samples -->
            </port>
        </output>
    </layer>

*Example 2: 2D input tensor with multiple batches.*

.. code-block:: xml
   :force:

    <layer ... name="Multinomial" type="Multinomial">
        <data convert_type="f32", with_replacement="true", log_probs="true", global_seed="234", op_seed="148"/>
        <input>
            <port id="0" precision="FP32">  <!-- probs value: [[-1, 1, 2], [50, 1, 21]] -->
                <dim>2</dim> <!-- batch size of 2 -->
                <dim>3</dim>
            </port>
            <port id="1" precision="I32"/> <!-- num_samples value: 10 -->
        </input>
        <output>
            <port id="3" precision="I32" names="Multinomial:0">
                <dim>2</dim> <!--dimension depends on input batch size -->
                <dim>10</dim> <!--dimension depends on num_samples -->
            </port>
        </output>
    </layer>

*Example 3: 2D input tensor without replacement.*

.. code-block:: xml
   :force:

    <layer ... name="Multinomial" type="Multinomial">
        <data convert_type="f32", with_replacement="false", log_probs="false", global_seed="234", op_seed="148"/>
        <input>
            <port id="0" precision="FP32">  <!-- probs value: [[0.1, 0.5, 0.4]] -->
                <dim>2</dim> <!-- batch size of 2 -->
                <dim>3</dim>
            </port>
            <port id="1" precision="I32"/> <!-- num_samples value: 2 -->
        </input>
        <output>
            <port id="3" precision="I32" names="Multinomial:0">
                <dim>2</dim> <!-- batch size of 2 -->
                <dim>2</dim> <!-- 2 unique samples of classes -->
            </port>
        </output>
    </layer>

