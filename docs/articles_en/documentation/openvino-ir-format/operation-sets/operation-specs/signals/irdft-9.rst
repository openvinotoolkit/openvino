Inverse Discrete complex-to-real Fourier Transformation (IRDFT)
===============================================================


.. meta::
  :description: Learn about IRDFT-9 - a signal processing operation, which can be
                performed on two required and one optional input tensor.

**Versioned name**: *IRDFT-9*

**Category**: *Signal processing*

**Short description**: *IRDFT* operation performs the inverse complex-to-real discrete Fourier transformation of the input tensor by specified dimensions.

**Attributes**:

No attributes available.

**Inputs**

* **1**: ``data`` - Input tensor of type *T* with data for the IRDFT transformation. The last dimension of the input tensor must be equal to 2, that is the input tensor shape must have the form ``[D_0, D_1, ..., D_{N-1}, 2]``, representing the real and imaginary components of complex numbers in ``[:, ..., :, 0]`` and in ``[:, ..., :, 1]`` correspondingly. **Required.**
* **2**: ``axes`` - 1D tensor of type *T_IND* specifying dimension indices where IRDFT is applied, and ``axes`` is any unordered list of indices of different dimensions of the input tensor, for example, ``[0, 4]``, ``[4, 0]``, ``[4, 2, 1]``, ``[1, 2, 3]``, ``[-3, 0, -2]``. These indices should be integers from ``-(r - 1)`` to ``(r - 2)`` inclusively, where ``r = rank(data)``. A negative axis ``a`` is interpreted as an axis ``r - 1 + a``. Other dimensions do not change. The order of elements in the ``axes`` attribute matters, and is mapped directly to elements in the third input ``signal_size``. **Required.**
*

  .. note::

     The following constraint must be satisfied: ``rank(data) >= len(axes) + 1 and (rank(data) - 1) not in axes and (-1) not in axes``.


* **3**: ``signal_size`` - 1D tensor of type *T_SIZE* describing signal size with respect to axes from the input ``axes``. If ``signal_size[i] == -1``, then IRDFT is calculated for full size of the axis ``axes[i]``. If ``signal_size[i] > data_shape[: r - 1][axes[i]]``, then input data is zero-padded with respect to the axis ``axes[i]`` at the end. Finally, if ``signal_size[i] < data_shape[: r - 1][axes[i]]``, then input data is trimmed with respect to the axis ``axes[i]``. More precisely, if ``signal_size[i] < data_shape[: r - 1][axes[i]]``, the slice ``0: signal_size[i]`` of the axis ``axes[i]`` is considered. Optionally, with default value ``[data_shape[: r - 1][a] for a in axes]``.
*

  .. note::

     If the input ``signal_size`` is specified, then the size of ``signal_size`` must be the same as the size of ``axes``.


**Outputs**

*   **1**: Resulting tensor with elements of the same type as input ``data`` tensor and with rank ``r - 1``, where ``r = rank(data)``. The shape of the output has the form ``[S_0, S_1, ..., S_{r-2}]``, where all ``S_a`` are calculated as follows:

1. Calculate ``normalized_axes``, where each ``normalized_axes[i] = axes[i]``, if ``axes[i] >= 0``, and ``normalized_axes[i] = axes[i] + r - 1`` otherwise.

2. If ``a not in normalized_axes``, then ``S_a = data_shape[a]``.

3. If ``a in normalized_axes``, then ``a = normalized_axes[i]`` for some ``i``. In such case, ``S_a = 2 * (data_shape[a] - 1)`` if the ``signal_size`` input is not specified, or, if it is specified, ``signal_size[i] = -1``; and ``S_a = signal_size[a]`` otherwise.
   + When ``i != len(normalized_axes) - 1``, ``S_a`` is calculated as ``S_a = data_shape[a]`` if the ``signal_size`` input is not specified, or, if it is specified, ``signal_size[i] = -1``; and ``S_a = signal_size[a]`` otherwise.
   + When ``i = len(normalized_axes) - 1``, ``S_a`` is calculated as ``S_a = 2 * (data_shape[a] - 1)`` if the ``signal_size`` input is not specified, or, if it is specified, ``signal_size[i] = -1``; and ``S_a = signal_size[a]`` otherwise.

**Types**

* *T*: any supported floating-point type.

* *T_IND*: ``int64`` or ``int32``.

* *T_SIZE*: ``int64`` or ``int32``.

**Detailed description**: *IRDFT* performs the discrete Fourier transformation of the input tensor, according to the following rules.

For simplicity, assume that an input tensor ``A`` has the shape ``[B_0, ..., B_{k-1}, M_0, ..., M_{q-1}, 2]``, ``axes=[k,...,k + q - 1]``, and ``signal_size=[S_0,...,S_{q-1}]``.

Let ``D`` be a value of the input tensor ``A``.

Next, put

.. math::

   X[j_0,\dots,j_{k-1},j_k,\dots,j_{k+q-1}]=D[j_0,\dots,j_{k-1},j_k,\dots,j_{k+q-1},0]+iD[j_0,\dots,j_{k-1},j_k,\dots,j_{k+q-1},1]


for all indices ``j_0,...,j_{k+q-1}``, where ``i`` is an imaginary unit, that is ``X`` is a complex tensor.

Define the complex tensor ``F`` with the shape ``[B_0, ..., B_{k-1}, 2 * (M_0 - 1), ..., 2 * (M_{q-1} - 1)]`` using the formula

.. math::

   F[j_0,\dots,j_{k-1},j_k,\dots,j_p,\dots,j_{k+q-1}] = \begin{cases}X[j_0,\dots,j_{k-1},j_k,\dots,j_p,\dots,j_{k+q-1}],\text{ when }j_p=0,\dots,M_p-1;\\ \overline{X[j_0,\dots,j_{k-1},j_k,\dots,2(M_{p-1} - 1) - j_p,\dots,j_{k+q-1}]},\text{ otherwise.}\end{cases}


Construct the complex tensor ``G`` with the shape ``[B_0, ..., B_{k-1}, S_0, ..., S_{q-1}]`` by the following way. If ``S_a > 2 * (M_a - 1)``, then the axis ``k + a`` of ``F`` will be padded by zeros; if ``S_a < 2 * (M_a - 1)``, then the axis ``k + a`` of ``F`` will be trimmed, that is, we will consider only the slice ``0: S_a`` of this axis; finally, if ``S_a = 2 * (M_a - 1)``, then we consider the full axis ``k + a`` of ``F``.

Let ``Y`` be a complex tensor with the shape ``[B_0, ..., B_{k-1}, S_0, ..., S_{q-1}]`` such that

.. math::

   Y[n_0,\dots,n_{k-1},m_0,\dots,m_{q-1}]=\frac{1}{\prod\limits_{b=0}^{q-1}S_b}\sum\limits_{j_0=0}^{S_0-1}\cdots\sum\limits_{j_{q-1}=0}^{S_{q-1}-1}X[n_0,\dots,n_{k-1},j_0,\dots,j_{q-1}]\exp\left(2\pi i\sum\limits_{b=0}^{q-1}\frac{m_bj_b}{S_b}\right)


for all indices ``n_0,...,n_{k-1}``, ``m_0,...,m_{q-1}``.

Finally, the result of the inverse discrete complex-to-real Fourier transform is a real part of the tensor `Y`.

Calculations for the generic case of axes and signal sizes are similar.

**Example**:

There is no ``signal_size`` input (4D input tensor):

.. code-block:: xml
   :force:

   <layer ... type="IRDFT" ... >
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>161</dim>
               <dim>161</dim>
               <dim>2</dim>
           </port>
           <port id="1">
               <dim>2</dim> <!-- [1, 2] -->
           </port>
       <output>
           <port id="2">
               <dim>1</dim>
               <dim>161</dim>
               <dim>320</dim>
           </port>
       </output>
   </layer>


There is no ``signal_size`` input (3D input tensor):

.. code-block:: xml
   :force:

   <layer ... type="IRDFT" ... >
       <input>
           <port id="0">
               <dim>161</dim>
               <dim>161</dim>
               <dim>2</dim>
           </port>
           <port id="1">
               <dim>2</dim> <!-- [0, 1] -->
           </port>
       <output>
           <port id="2">
               <dim>161</dim>
               <dim>320</dim>
           </port>
       </output>
   </layer>


There is ``signal_size`` input (4D input tensor):

.. code-block:: xml
   :force:

   <layer ... type="IRDFT" ... >
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>161</dim>
               <dim>161</dim>
               <dim>2</dim>
           </port>
           <port id="1">
               <dim>2</dim> <!-- [1, 2] -->
           </port>
           <port id="2">
               <dim>2</dim> <!-- [512, 100] -->
           </port>
       <output>
           <port id="3">
               <dim>1</dim>
               <dim>512</dim>
               <dim>100</dim>
           </port>
       </output>
   </layer>



There is ``signal_size`` input (3D input tensor):

.. code-block:: xml
   :force:

   <layer ... type="IRDFT" ... >
       <input>
           <port id="0">
               <dim>161</dim>
               <dim>161</dim>
               <dim>2</dim>
           </port>
           <port id="1">
               <dim>2</dim> <!-- [0, 1] -->
           </port>
           <port id="2">
               <dim>2</dim> <!-- [512, 100] -->
           </port>
       <output>
           <port id="3">
               <dim>512</dim>
               <dim>100</dim>
           </port>
       </output>
   </layer>



There is ``signal_size`` input (5D input tensor, ``-1`` in ``signal_size``, unsorted axes):

.. code-block:: xml
   :force:

   <layer ... type="IRDFT" ... >
       <input>
           <port id="0">
               <dim>16</dim>
               <dim>768</dim>
               <dim>580</dim>
               <dim>320</dim>
               <dim>2</dim>
           </port>
           <port id="1">
               <dim>3</dim> <!-- axes input contains  [3, 1, 2] -->
           </port>
           <port id="2">
               <dim>3</dim> <!-- signal_size input contains [170, -1, 1024] -->
           </port>
       <output>
           <port id="3">
               <dim>16</dim>
               <dim>768</dim>
               <dim>1024</dim>
               <dim>170</dim>
           </port>
       </output>
   </layer>


There is ``signal_size`` input (5D input tensor, ``-1`` in ``signal_size``, unsorted axes, the second example):

.. code-block:: xml
   :force:

   <layer ... type="IRDFT" ... >
       <input>
           <port id="0">
               <dim>16</dim>
               <dim>768</dim>
               <dim>580</dim>
               <dim>320</dim>
               <dim>2</dim>
           </port>
           <port id="1">
               <dim>3</dim> <!-- axes input contains  [3, 0, 2] -->
           </port>
           <port id="2">
               <dim>3</dim> <!-- signal_size input contains [258, -1, 2056] -->
           </port>
       <output>
           <port id="3">
               <dim>16</dim>
               <dim>768</dim>
               <dim>2056</dim>
               <dim>258</dim>
           </port>
       </output>
   </layer>



