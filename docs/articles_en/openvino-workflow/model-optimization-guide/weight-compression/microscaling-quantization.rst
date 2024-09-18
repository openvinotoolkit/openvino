Microscaling (MX) Quantization
==============================

Microscaling (MX) Quantization method has been introduced to enable users
to quantize LLMs with a high compression rate at minimal cost of accuracy.
The method helps maintain model performance comparable to that of the conventional
FP32. It increases compute and storage efficiency by using low bit-width
floating point and integer-based data formats:

+---------------+-----------------+----------------------------+
|  Data format  |  Data type      | Description                |
+===============+=================+============================+
| MXFP8         | | FP8 (E5M2)    | | Floating point, 8-bit    |
|               | | FP8 (E4M3)    | | Floating point, 8-bit    |
+---------------+-----------------+----------------------------+
| MXFP6         | | FP6 (E3M2)    | | Floating point, 6-bit    |
|               | | FP6 (E2M3)    | | Floating point, 6-bit    |
+---------------+-----------------+----------------------------+
| **MXFP4**     | **FP4 (E2M1)**  | **Floating point, 4-bit**  |
+---------------+-----------------+----------------------------+
| MXINT8        | INT8            | Integer, 8-bit             |
+---------------+-----------------+----------------------------+


.. _mxfp4_support:

**Currently, only the**
`MXFP4 (E2M1) <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
**data format is supported in NNCF and for quantization on CPU.**
E2M1 may be considered for improving accuracy, however, quantized models will
not be faster than the ones compressed to INT8_ASYM.

Quantization to the E2M1 data type will compress weights to 4-bit without a zero
point and with 8-bit E8M0 scales. To quantize a model to E2M1, set
``mode=CompressWeightsMode.E2M1`` in ``nncf.compress_weights()``. It is
recommended to use ``group size = 32``. See the example below:

.. code-block:: py

   from nncf import compress_weights, CompressWeightsMode
   compressed_model = compress_weights(model, mode=CompressWeightsMode.E2M1, group_size=32, all_layers=True)

.. note::

   Different values for ``group_size`` and ``ratio`` are also supported.


Additional Resources
####################

* `OCP Microscaling Formats (MX) Specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
* `Intel® Neural Compressor Documentation <https://intel.github.io/neural-compressor/latest/docs/source/3x/PT_MXQuant.html>`__