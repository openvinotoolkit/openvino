RandomPoisson
=============


.. meta::
  :description: Learn about RandomPoisson-17 - a generation operation that samples
                from a Poisson distribution for each rate in the input tensor.

**Versioned name**: *RandomPoisson-17*

**Category**: *Generation*

**Short description**: *RandomPoisson* generates a tensor of the same shape and type as
the input, where each element is sampled from a Poisson distribution with rate given by
the corresponding input element.

**Detailed description**:

*RandomPoisson* draws one Poisson sample per input rate:

.. math::

   \mathrm{out}_i \sim \mathrm{Poisson}(\mathrm{input}_i)

The input tensor must contain non-negative rates (including zero). Negative rates and
NaN are invalid. A rate of zero yields output zero. Scalar (rank-0) inputs are not
supported.

This I/O matches `torch.poisson <https://docs.pytorch.org/docs/2.13/generated/torch.poisson.html>`__:
a rates tensor in, a same-shape tensor out, one sample per element. TensorFlow
`tf.random.poisson <https://www.tensorflow.org/api_docs/python/tf/random/poisson>`__ /
`RandomPoissonV2 <https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/random-poisson-v2>`__
instead take a separate ``shape`` and a rate tensor ``lam`` that can broadcast so many
independent samples are drawn per rate. OpenVINO does not use that shape/broadcast API.
``alignment`` only selects which framework’s **uniform stream** (and skip pattern) to
follow when converting those uniforms into Poisson counts.

This op uses two algorithms, depending on rate. If rate >= 10, then the algorithm by
Hormann is used to acquire samples via transformation-rejection. See
`http://www.sciencedirect.com/science/article/pii/0167668793909974 <http://www.sciencedirect.com/science/article/pii/0167668793909974>`__.

Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform random
variables. See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
Programming, Volume 2. Addison Wesley.

Both PyTorch (`sample_poisson` in ATen) and TensorFlow RandomPoissonV2 use the same
**algorithm split** on the rate :math:`\lambda`:

* If :math:`\lambda = 0`, the sample is ``0``.
* If :math:`0 < \lambda < 10`, Knuth’s method (product of uniforms vs. :math:`e^{-\lambda}`).
* If :math:`\lambda \ge 10`, Hörmann’s transformed rejection method (1993).

Rate arithmetic (``exp``, ``log``, ``lgamma``, and the Knuth product) is performed in
``double``. The integer count is then cast back to the input element type
(``bf16``, ``f16``, ``f32``, or ``f64``).

If both *global_seed* and *op_seed* are zero, *RandomPoisson* generates a
non-deterministic sequence. Plugin implementations may differ in that case.

**Uniform source and ``alignment``**:

Uniforms on :math:`[0, 1)` are produced with the same generators as
*RandomUniform-8*:

* ``alignment = TENSORFLOW`` (default): Philox (4x32_10). See *RandomUniform-8* for the
  Philox round structure. TensorFlow RandomPoissonV2 advances a **separate Philox
  substream per output index** by skipping ``256 * i`` 128-bit Philox results from the
  shared starting state (``256`` is TensorFlow’s
  ``kReservedSamplesPerOutput``). Leftover uniforms in a Philox block are consumed in
  reverse order, matching TensorFlow’s ``Uniform`` helper.
* ``alignment = PYTORCH``: Mersenne-Twister (MT19937), same generator family as
  *RandomUniform-8* with PyTorch alignment. Uniforms are consumed **forward** from a
  single sequential stream. *op_seed* is ignored by the generator (only *global_seed*
  seeds the state), as for *RandomUniform-8*.

**Knuth algorithm** (Donald E. Knuth (1969). Seminumerical Algorithms. The Art of
Computer Programming, Volume 2. Addison Wesley):

Let :math:`U_j` be i.i.d. uniforms on :math:`(0, 1]`. Set :math:`p \leftarrow 1`,
:math:`k \leftarrow -1`. Repeat :math:`p \leftarrow p \cdot U`, :math:`k \leftarrow k + 1`
until :math:`p \le e^{-\lambda}`. Return :math:`k`.

**Hörmann transformed rejection**
(`Hormann, transformation-rejection <http://www.sciencedirect.com/science/article/pii/0167668793909974>`__):

Let :math:`s = \sqrt{\lambda}`, :math:`b = 0.931 + 2.53 s`, :math:`a = -0.059 + 0.02483 b`,
:math:`\alpha^{-1} = 1.1239 + 1.1328 / (b - 3.4)`, :math:`v_r = 0.9277 - 3.6224 / (b - 2)`.
Draw uniforms :math:`U, V`. Set :math:`u = U - 0.5`, :math:`u_s = 0.5 - |u|`, and

.. math::

   k = \left\lfloor \left(\frac{2a}{u_s} + b\right) u + \lambda + 0.43 \right\rfloor

Accept :math:`k` on the cheap squeeze :math:`u_s \ge 0.07` and :math:`V \le v_r`, reject
invalid :math:`k < 0` or a thin-tail probe, otherwise accept using the log-gamma
comparison against :math:`-\lambda + k \log \lambda - \log \Gamma(k+1)`. Repeat until
acceptance.

**Example 1.** *RandomPoisson* with ``global_seed = 150``, ``op_seed = 69``,
``alignment = PYTORCH``, input type ``f32`` (Knuth rates):

.. code-block:: cpp
   :force:

   input  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
   output = [1, 2, 2, 2, 3, 8, 9, 11, 5]


**Example 2.** *RandomPoisson* with ``global_seed = 150``, ``op_seed = 69``,
``alignment = PYTORCH``, input type ``f32`` (Hörmann rates):

.. code-block:: cpp
   :force:

   input  = [11, 12, 13, 14, 15, 16, 17, 18, 19]
   output = [14, 11, 15, 20, 9, 20, 15, 17, 21]


**Example 3.** *RandomPoisson* with ``global_seed = 150``, ``op_seed = 77``,
``alignment = TENSORFLOW``, input type ``f32`` (mixed rates: zero, Knuth, and Hörmann):

.. code-block:: cpp
   :force:

   input  = [0, 5, 10, 15, 20, 25, 30, 35, 40]
   output = [0, 4, 5, 12, 21, 20, 38, 39, 34]


**Attributes**:

* ``global_seed``

  * **Description**: global seed value.
  * **Range of values**: non-negative integers
  * **Type**: `unsigned int 64-bit`
  * **Default value**: 0
  * **Required**: *No*

* ``op_seed``

  * **Description**: operational seed value. Used with TensorFlow (Philox) alignment.
    Ignored by the PyTorch (Mersenne-Twister) generator.
  * **Range of values**: non-negative integers
  * **Type**: `unsigned int 64-bit`
  * **Default value**: 0
  * **Required**: *No*

* ``alignment``

  * **Description**: framework whose uniform generation (and TensorFlow skip pattern)
    the Poisson sampler should match.
  * **Range of values**: TENSORFLOW, PYTORCH
  * **Type**: `string`
  * **Default value**: TENSORFLOW
  * **Required**: *No*

**Inputs**:

* **1**: ``input`` - tensor of type *T* with Poisson rates (non-negative). Shape rank
  is at least 1. **Required.**

**Outputs**:

* **1**: A tensor of type *T* and the same shape as ``input``. Values are integer Poisson
  counts represented in *T*.

**Types**

* *T*: ``bf16``, ``f16``, ``f32``, or ``f64``.

*Example 1: 1D rates, PyTorch alignment, ``f32``.*

.. code-block:: xml
   :force:

    <layer ... name="RandomPoisson" type="RandomPoisson">
        <data global_seed="150" op_seed="69" alignment="pytorch"/>
        <input>
            <port id="0" precision="FP32">  <!-- rates: [1, 2, 3, 4, 5, 6, 7, 8, 9] -->
                <dim>9</dim>
            </port>
        </input>
        <output>
            <port id="1" precision="FP32" names="RandomPoisson:0">  <!-- [1, 2, 2, 2, 3, 8, 9, 11, 5] -->
                <dim>9</dim>
            </port>
        </output>
    </layer>

*Example 2: 1D rates, TensorFlow alignment, ``f16``.*

.. code-block:: xml
   :force:

    <layer ... name="RandomPoisson" type="RandomPoisson">
        <data global_seed="150" op_seed="77" alignment="tensorflow"/>
        <input>
            <port id="0" precision="FP16">  <!-- rates: [1, 2, 3, 4, 5, 6, 7, 8, 9] -->
                <dim>9</dim>
            </port>
        </input>
        <output>
            <port id="1" precision="FP16" names="RandomPoisson:0">  <!-- [1, 1, 1, 2, 9, 5, 3, 3, 8] -->
                <dim>9</dim>
            </port>
        </output>
    </layer>

*Example 3: 2D rates, TensorFlow alignment (default). Output shape matches input.*

.. code-block:: xml
   :force:

    <layer ... name="RandomPoisson" type="RandomPoisson">
        <data global_seed="150" op_seed="77" alignment="tensorflow"/>
        <input>
            <port id="0" precision="FP32">  <!-- rates: [[0, 5, 10], [15, 20, 25], [30, 35, 40]] -->
                <dim>3</dim>
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="1" precision="FP32" names="RandomPoisson:0">  <!-- [[0, 4, 5], [12, 21, 20], [38, 39, 34]] -->
                <dim>3</dim>
                <dim>3</dim>
            </port>
        </output>
    </layer>

*Example 4: 1D rates, PyTorch alignment, ``bf16``.*

.. code-block:: xml
   :force:

    <layer ... name="RandomPoisson" type="RandomPoisson">
        <data global_seed="150" op_seed="69" alignment="pytorch"/>
        <input>
            <port id="0" precision="BF16">  <!-- rates: [0, 5, 10, 15, 20, 25, 30, 35, 40] -->
                <dim>9</dim>
            </port>
        </input>
        <output>
            <port id="1" precision="BF16" names="RandomPoisson:0">  <!-- [0, 6, 6, 11, 19, 15, 36, 33, 33] -->
                <dim>9</dim>
            </port>
        </output>
    </layer>
