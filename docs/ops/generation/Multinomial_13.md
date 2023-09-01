# Multinomial {#openvino_docs_ops_generation_Multinomial_13}

@sphinxdirective

.. meta::
  :description: Learn about Multinomial-13 - a generation operation, that creates sequence of indices of sampled classes from Multinomial distribution.

**Versioned name**: *Multinomial-13*

**Category**: *Generation*

**Short description**: *Multinomial* operation generates a sequence of class indices sampled from Multinomial distribution.

**Detailed description**: *Multinomial* operation generates a sequence of class indices sampled from Multinomial distribution.

Values in *input* indicate probabilities for every class that could be randomly sampled from Multinominal distribution. When specific class is sampled, it's indice in *input* is appended to *output* sequence in corresponding batch value.

**Attributes**:

* ``output_type``

  * **Description**: the type of the output. Determines generation algorithm and affects resulting values. Output numbers generated for different values of *output_type* may not be equal.
  * **Range of values**: "i32", "i64", "f16", "bf16", "f32", "f64".
  * **Type**: string
  * **Required**: *Yes*

* ``replacement``

  * **Description**: allows to control wether classes could repeat in sampled sequence.
  * **Range of values**: `true`, `false`
      * ``true`` - class indices can repeat in sampled sequence.
      * ``false`` - class indices will not repeat in sequence and size of *input* ``class_size`` dimension is required to be larger or equal of *num_samples* value.
  * **Type**: `bool`
  * **Required**: *Yes*

* ``log_probs``

  * **Description**: allows to control wether *inputs* should be treated as probabilities or unnormalized log probabilities.
  * **Range of values**: `true`, `false`
    * ``true`` - set values in *inputs* are unnormalized log probabilites that can be any real number.
    * ``false`` - probabilities in *inputs* are expected to be non-negative, finite and have non-zero sum.
  * **Type**: `bool`
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

*   **1**: ``input`` - 1D or 2D tensor of type `T_IN` and shape `[class_size]` or `[batch_size, class_size]` containing probabilities. Allowed values depend on *log_probs* attribute and are internally normalized to have values in range of `[0, 1]`, sum of all probabilities in given batch is equal to 1. **Required.**

*   **2**: ``num_samples`` - scalar or 1D tensor with 1 element of type `T_SAMPLES` specifying number of samples to draw from Multinomial distribution. **Required.**

**Outputs**:

* **1**:  ``output`` A tensor with type specified by the attribute *output_type* and shape depending on *input* and *num_samples*, either ``[num_samples]`` or ``[batch_size, num_samples]``.

**Types**

* **T_IN**: any supported floating-point type.
* **T_SAMPLES**: any supported integer type.


*Example 1: 1D input tensor.*

.. code-block:: xml
   :force:

    <layer ... name="Multinomial" type="Multinomial">
        <data output_type="f32" replacement="true", log_probs="false" global_seed="234" op_seed="148"/>
        <input>
            <port id="0" precision="FP32">  < !-- shape value: [0.1, 0.5, 0.4] -->
                <dim>3</dim>
            </port>
            <port id="1" precision="I32"/> < !-- num_samples value: 5 -->
        </input>
        <output>
            <port id="3" precision="FP32" names="Multinomial:0">
                <dim>5</dim>
            </port>
        </output>
    </layer>

*Example 2: 2D input tensor.*

.. code-block:: xml
   :force:

    <layer ... name="Multinomial" type="Multinomial">
        <data output_type="f32" replacement="true", log_probs="true" global_seed="234" op_seed="148"/>
        <input>
            <port id="0" precision="FP32">  < !-- shape value: [50, 1, 21] -->
                <dim>16</dim> < !-- batch size of 16 -->
                <dim>3</dim>
            </port>
            <port id="1" precision="I32"/> < !-- num_samples value: 8 -->
        </input>
        <output>
            <port id="3" precision="FP32" names="Multinomial:0">
                <dim>16</dim> < !--dimension depends on input batch size -->
                <dim>8</dim> < !--dimension depends on num_samples -->
            </port>
        </output>
    </layer>

*Example 3: 1D input tensor without replacement.*

.. code-block:: xml
   :force:

    <layer ... name="Multinomial" type="Multinomial">
        <data output_type="f32" replacement="false", log_probs="false" global_seed="234" op_seed="148"/>
        <input>
            <port id="0" precision="FP32">  < !-- shape value: [0.1, 0.5, 0.4] -->
                <dim>3</dim>
            </port>
            <port id="1" precision="I32"/> < !-- num_samples value: 2 -->
        </input>
        <output>
            <port id="3" precision="FP32" names="Multinomial:0">
                <dim>2</dim> < !-- 2 unique samples of classes -->
            </port>
        </output>
    </layer>

@endsphinxdirective
