Histc
=====


.. meta::
  :description: Learn about Histc-17 - a reduction operation, which computes
                the histogram of all elements of a single input tensor.

**Versioned name**: *Histc-17*

**Category**: *Reduction*

**Short description**: *Histc* computes the histogram of all elements of the input tensor using equal-width bins over a given range.

**Detailed description**

*Histc* is compatible with ``torch.histc`` from PyTorch. The input tensor is treated as a flat collection of values, regardless of its rank. The range ``[lo, hi]`` is divided into *bins* bins of equal width. The operation returns a 1D tensor with the number of input elements that fall into each bin.

The operation is performed in the following steps:

1. Selecting the range ``[lo, hi]``:

   * ``lo = min`` and ``hi = max``.
   * If ``lo == hi`` and the input tensor is not empty, the range is taken from the data: ``lo`` is the minimum and ``hi`` is the maximum of all input elements. Since the default values of both *min* and *max* are ``0``, the range is computed from the data by default.
   * If ``lo == hi`` still holds (all input elements are equal or the input tensor is empty), the range is widened: ``lo = lo - 1`` and ``hi = hi + 1``. For *T* equal to ``f32`` or ``f64`` the new bounds are ``lo = min(nextafter(lo, -inf), lo - 1)`` and ``hi = max(nextafter(hi, +inf), hi + 1)``, where ``nextafter`` is evaluated in type *T*. This is the only difference from plain ``lo - 1`` and ``hi + 1``, and it is visible only for values of large magnitude.

2. The resulting range must be finite and satisfy ``lo < hi``.

3. Counting: Each input element ``x`` contributes ``1`` to exactly one bin, or is ignored:

   * Elements with ``x < lo``, ``x > hi``, and ``NaN`` elements are ignored.
   * Any other element goes to the bin with the index ``idx = floor((x - lo) * bins / (hi - lo))``, where the result of the division is truncated toward zero. Bins are half-open ``[lo + i * w, lo + (i + 1) * w)``, where ``w = (hi - lo) / bins``, except for the last bin, which is closed and includes ``hi``. This is enforced by ``idx = bins - 1`` whenever ``x == hi``.

   * The bin index is computed in type *T*, so elements lying very close to a bin boundary may be assigned to a neighboring bin due to rounding. This matches the behavior of PyTorch, which does not guarantee consistency with exact bin boundaries.

The output element ``output[i]`` is the number of elements assigned to the bin ``i``, represented in type *T*. For floating-point types of low precision (for example ``f16`` or ``bf16``) counts that cannot be represented exactly in *T* are rounded.

**Attributes**

* *bins*

  * **Description**: number of equal-width histogram bins. It determines the shape of the output.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: ``100``
  * **Required**: *no*

* *min*

  * **Description**: lower end of the histogram range (inclusive). If *min* and *max* are equal, the range is taken from the data as described above.
  * **Range of values**: a finite floating-point number
  * **Type**: ``double``
  * **Default value**: ``0``
  * **Required**: *no*

* *max*

  * **Description**: upper end of the histogram range (inclusive). If *min* and *max* are equal, the range is taken from the data as described above.
  * **Range of values**: a finite floating-point number, not less than *min*
  * **Type**: ``double``
  * **Default value**: ``0``
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - a tensor of type *T* and arbitrary shape, including a scalar or an empty tensor. **Required.**

**Outputs**

* **1**: A 1D tensor of type *T* and shape ``[bins]``, containing the number of input elements in each bin. If the input tensor is empty, all elements of the output are ``0``.

**Types**

* *T*: any supported floating-point or integer type.

**Examples**

*Range given explicitly*

``data = [1, 2, 1]``, ``bins = 4``, ``min = 0``, ``max = 3``. The bin width is ``0.75``, so the bins are ``[0, 0.75)``, ``[0.75, 1.5)``, ``[1.5, 2.25)``, ``[2.25, 3]``. The output is ``[0, 2, 1, 0]``.

.. code-block:: xml
   :force:

   <layer ... type="Histc" version="opset17">
       <data bins="4" min="0" max="3"/>
       <input>
           <port id="0" precision="FP32">
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="1" precision="FP32">
               <dim>4</dim>
           </port>
       </output>
   </layer>

*Range taken from the data*

``data = [[1, 2], [1, 4]]``, ``bins = 4``, ``min = 0``, ``max = 0``. The range is ``[1, 4]`` and the bin width is ``0.75``. The value ``4`` equals ``hi``, so it falls into the last bin. The output is ``[2, 1, 0, 1]``.

.. code-block:: xml
   :force:

   <layer ... type="Histc" version="opset17">
       <data bins="4" min="0" max="0"/>
       <input>
           <port id="0" precision="FP32">
               <dim>2</dim>
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="1" precision="FP32">
               <dim>4</dim>
           </port>
       </output>
   </layer>
