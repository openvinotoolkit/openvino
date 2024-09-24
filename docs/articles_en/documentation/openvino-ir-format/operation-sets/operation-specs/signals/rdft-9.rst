Discrete Fourier Transformation for real-valued input (RDFT)
============================================================


.. meta::
  :description: Learn about RDFT-9 - a signal processing operation, which can be
                performed on two required and one optional input tensor.

**Versioned name**: *RDFT-9*

**Category**: *Signal processing*

**Short description**: *RDFT* operation performs the discrete real-to-complex Fourier transformation of the input tensor by specified dimensions.

**Attributes**:

No attributes available.

**Inputs**

*   **1**: ``data`` - Input tensor of type *T* with data for the RDFT transformation. **Required.**
*   **2**: ``axes`` - 1D tensor of type *T_IND* specifying dimension indices where RDFT is applied, and ``axes`` is any unordered list of indices of different dimensions of input tensor, for example, ``[0, 4]``, ``[4, 0]``, ``[4, 2, 1]``, ``[1, 2, 3]``, ``[-3, 0, -2]``. These indices should be integers from ``-r`` to ``r - 1`` inclusively, where ``r = rank(data)``. A negative axis ``a`` is interpreted as an axis ``r + a``. Other dimensions do not change. The order of elements in ``axes`` attribute matters, and is mapped directly to elements in the third input ``signal_size``. **Required.**
*   **3**: ``signal_size`` - 1D tensor of type *T_SIZE* describing signal size with respect to axes from the input ``axes``. If ``signal_size[i] == -1``, then RDFT is calculated for full size of the axis ``axes[i]``. If ``signal_size[i] > data_shape[axes[i]]``, then input data is zero-padded with respect to the axis ``axes[i]`` at the end. Finally, ``signal_size[i] < data_shape[axes[i]]``, then input data is trimmed with respect to the axis ``axes[i]``. More precisely, if ``signal_size[i] < data_shape[axes[i]]``, the slice ``0: signal_size[i]`` of the axis ``axes[i]`` is considered. Optionally, with default value ``[data_shape[a] for a in axes]``.
*   **NOTE**: If the input ``signal_size`` is specified, the size of ``signal_size`` must be the same as the size of ``axes``.

**Outputs**

*   **1**: Resulting tensor with elements of the same type as input ``data`` tensor and with rank ``r + 1``, where ``r = rank(data)``. The shape of the output has the form ``[S_0, S_1, ..., S_{r-1}, 2]``, where all ``S_a`` are calculated as follows:

1. Calculate ``normalized_axes``, where each ``normalized_axes[i] = axes[i]``, if ``axes[i] >= 0``, and ``normalized_axes[i] = axes[i] + r`` otherwise.

2. If ``a not in normalized_axes``, then ``S_a = data_shape[a]``.

3. If ``a in normalized_axes``, then ``a = normalized_axes[i]`` for some ``i``.

   + When ``i != len(normalized_axes) - 1``, ``S_a`` is calculated as ``S_a = data_shape[a]`` if the ``signal_size`` input is not specified, or, if it is specified, ``signal_size[i] = -1``; and ``S_a = signal_size[a]`` otherwise.
   + When ``i = len(normalized_axes) - 1``, ``S_a`` is calculated as ``S_a = data_shape[a] // 2 + 1`` if the ``signal_size`` input is not specified, or, if it is specified, ``signal_size[i] = -1``; and ``S_a = signal_size[a] // 2 + 1`` otherwise.

**Types**

* *T*: any supported floating-point type.

* *T_IND*: ``int64`` or ``int32``.

* *T_SIZE*: ``int64`` or ``int32``.

**Detailed description**: *RDFT* performs the discrete Fourier transformation of real-valued input tensor with respect to specified axes. Calculations are performed according to the following rules.

For simplicity, assume that an input tensor ``A`` has the shape ``[B_0, ..., B_{k-1}, M_0, ..., M_{q-1}]``, ``axes=[k,...,k+q-1]``, and ``signal_size=[S_0,...,S_{q-1}]``.

Let ``D`` be an input tensor ``A``, taking into account the ``signal_size``, and, hence, ``D`` has the shape ``[B_0, ..., B_{k-1}, S_0, ..., S_{q-1}]``.

Next, let

.. math::

	X=X[j_0,\dots,j_{k-1},j_k,\dots,j_{k+q-1}]

for all indices ``j_0,...,j_{k+q-1}``, be a real-valued input tensor.

Then the transformation RDFT of the tensor ``X`` is the tensor ``Y`` of the shape ``[B_0, ..., B_{k-1}, S_0 // 2 + 1, ..., S_{r-1} // 2 + 1]``, such that

.. math::

	Y[n_0,\dots,n_{k-1},m_0,\dots,m_{q-1}]=\sum\limits_{j_0=0}^{S_0-1}\cdots\sum\limits_{j_{q-1}=0}^{S_{q-1}-1}X[n_0,\dots,n_{k-1},j_0,\dots,j_{q-1}]\exp\left(-2\pi i\sum\limits_{b=0}^{q-1}\frac{m_bj_b}{S_b}\right)

for all indices ``n_0,...,n_{k-1}``, ``m_0,...,m_{q-1}``.

Calculations for the generic case of axes and signal sizes are similar.

**Example**:

There is no ``signal_size`` input (3D input tensor):

.. code-block:: xml
   :force:

    <layer ... type="RDFT" ... >
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>320</dim>
                <dim>320</dim>
            </port>
            <port id="1">
                <dim>2</dim> <!-- axes input contains [1, 2] -->
            </port>
        <output>
            <port id="2">
                <dim>1</dim>
                <dim>320</dim>
                <dim>161</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>


There is no ``signal_size`` input (2D input tensor):

.. code-block:: xml
   :force:

    <layer ... type="RDFT" ... >
        <input>
            <port id="0">
                <dim>320</dim>
                <dim>320</dim>
            </port>
            <port id="1">
                <dim>2</dim> <!-- axes input contains [0, 1] -->
            </port>
        <output>
            <port id="2">
                <dim>320</dim>
                <dim>161</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>



There is ``signal_size`` input (3D input tensor):

.. code-block:: xml
   :force:

    <layer ... type="RDFT" ... >
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>320</dim>
                <dim>320</dim>
            </port>
            <port id="1">
                <dim>2</dim> <!-- axes input contains [1, 2] -->
            </port>
            <port id="2">
                <dim>2</dim> <!-- signal_size input contains [512, 100] -->
            </port>
        <output>
            <port id="3">
                <dim>1</dim>
                <dim>512</dim>
                <dim>51</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>

There is ``signal_size`` input (2D input tensor):

.. code-block:: xml
   :force:

    <layer ... type="RDFT" ... >
        <input>
            <port id="0">
                <dim>320</dim>
                <dim>320</dim>
            </port>
            <port id="1">
                <dim>2</dim> <!-- axes input contains [0, 1] -->
            </port>
            <port id="2">
                <dim>2</dim> <!-- signal_size input contains [512, 100] -->
            </port>
        <output>
            <port id="3">
                <dim>512</dim>
                <dim>51</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>


There is ``signal_size`` input (4D input tensor, ``-1`` in ``signal_size``, unsorted axes):

.. code-block:: xml
   :force:

    <layer ... type="RDFT" ... >
        <input>
            <port id="0">
                <dim>16</dim>
                <dim>768</dim>
                <dim>580</dim>
                <dim>320</dim>
            </port>
            <port id="1">
                <dim>3</dim> <!-- axes input contains  [3, 1, 2] -->
            </port>
            <port id="2">
                <dim>3</dim> <!-- signal_size input contains [170, -1, 1024] -->
            </port>
        <output>
            <port id="3">
                <dim>16</dim>
                <dim>768</dim>
                <dim>513</dim>
                <dim>170</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>

There is ``signal_size`` input (4D input tensor, ``-1`` in ``signal_size``, unsorted axes, the second example):

.. code-block:: xml
   :force:

    <layer ... type="RDFT" ... >
        <input>
            <port id="0">
                <dim>16</dim>
                <dim>768</dim>
                <dim>580</dim>
                <dim>320</dim>
            </port>
            <port id="1">
                <dim>3</dim> <!-- axes input contains  [3, 0, 2] -->
            </port>
            <port id="2">
                <dim>3</dim> <!-- signal_size input contains [258, -1, 2056] -->
            </port>
        <output>
            <port id="3">
                <dim>16</dim>
                <dim>768</dim>
                <dim>1029</dim>
                <dim>258</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>

