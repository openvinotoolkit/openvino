.. {#openvino_docs_ops_internal_GGUFDequantize}

GGUFDequantize
==============


.. meta::
  :description: Learn about GGUFDequantize - an operation that decodes GGUF k-quant weight
                blocks into a floating-point tensor.

**Versioned name**: *GGUFDequantize*

**Category**: *Internal*

**Short description**: *GGUFDequantize* decodes GGUF k-quant weight blocks, exactly as stored
in a GGUF file, into a floating-point tensor.

**Detailed description**: GGUF (`format <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>`__)
stores every row of a quantized weight as a sequence of blocks of ``256`` values. Each block carries
its own scales, so the bytes cannot be expressed as a low-precision element type with separate scale
and zero-point tensors without converting them. *GGUFDequantize* takes the block bytes unchanged and
produces the decoded values. The block layouts and the decoding below are those of ggml
(`ggml-common.h <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h>`__,
``dequantize_row_q4_K``, ``dequantize_row_q5_K`` and ``dequantize_row_q6_K`` in
`ggml-quants.c <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c>`__).

Every row of ``B`` bytes is ``B / block_bytes`` consecutive blocks, and every block decodes to ``256``
consecutive values of the output row, so an output row has ``K = B / block_bytes * 256`` values.
All multi-byte fields are little-endian, ``f16`` fields are IEEE 754 half-precision values, and the
values are computed in ``f32`` and converted to *T*.

*Q4_K* and *Q5_K* blocks have 8 sub-blocks of 32 values, each with a 6-bit scale and a 6-bit minimum,
packed into 12 bytes:

.. code-block:: py

   def scale_min(s, scales):  # scales: the 12 packed bytes, s = 0 .. 7
       if s < 4:
           return scales[s] & 63, scales[s + 4] & 63
       return ((scales[s + 4] & 0xF) | ((scales[s - 4] >> 6) << 4),
               (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4))

* *Q4_K*: ``block_bytes = 144``.

  =========  ======  ================================================
  Bytes      Field   Content
  =========  ======  ================================================
  0 .. 1     d       ``f16`` super-block scale
  2 .. 3     dmin    ``f16`` super-block minimum
  4 .. 15    scales  6-bit scales and minimums of the 8 sub-blocks
  16 .. 143  qs      4-bit quants, two per byte
  =========  ======  ================================================

* *Q5_K*: ``block_bytes = 176``, the *Q4_K* fields plus the 5th bit of every quant.

  =========  ======  ================================================
  Bytes      Field   Content
  =========  ======  ================================================
  0 .. 1     d       ``f16`` super-block scale
  2 .. 3     dmin    ``f16`` super-block minimum
  4 .. 15    scales  6-bit scales and minimums of the 8 sub-blocks
  16 .. 47   qh      5th bit of every quant
  48 .. 175  qs      low 4 bits of every quant, two per byte
  =========  ======  ================================================

The 256 values ``y`` of a *Q4_K* or *Q5_K* block (``qh`` bits are ``0`` for *Q4_K*):

.. code-block:: py

   for j in range(4):  # sub-block 2 * j: low 4 bits, sub-block 2 * j + 1: high 4 bits
       sc0, m0 = scale_min(2 * j, scales)
       sc1, m1 = scale_min(2 * j + 1, scales)
       for l in range(32):
           q0 = (qs[32 * j + l] & 0xF) + 16 * ((qh[l] >> (2 * j)) & 1)
           q1 = (qs[32 * j + l] >> 4) + 16 * ((qh[l] >> (2 * j + 1)) & 1)
           y[64 * j + l] = d * sc0 * q0 - dmin * m0
           y[64 * j + 32 + l] = d * sc1 * q1 - dmin * m1

* *Q6_K*: ``block_bytes = 210``, 16 sub-blocks of 16 values with a signed 8-bit scale each.

  ===========  ======  ==============================================
  Bytes        Field   Content
  ===========  ======  ==============================================
  0 .. 127     ql      low 4 bits of every quant, two per byte
  128 .. 191   qh      high 2 bits of every quant, four per byte
  192 .. 207   scales  ``int8`` scales of the 16 sub-blocks
  208 .. 209   d       ``f16`` super-block scale
  ===========  ======  ==============================================

The 256 values ``y`` of a *Q6_K* block:

.. code-block:: py

   for n in range(2):  # two halves of 128 values
       L, H, S = ql[64 * n:], qh[32 * n:], scales[8 * n:]
       for l in range(32):
           s = l // 16
           q0 = ((L[l] & 0xF) | (((H[l] >> 0) & 3) << 4)) - 32
           q1 = ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32
           q2 = ((L[l] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32
           q3 = ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32
           y[128 * n + l] = d * S[s] * q0
           y[128 * n + l + 32] = d * S[s + 2] * q1
           y[128 * n + l + 64] = d * S[s + 4] * q2
           y[128 * n + l + 96] = d * S[s + 6] * q3


**Attributes**

* *format*

  * **Description**: the GGUF block format of the input.
  * **Range of values**: ``Q4_K``, ``Q5_K``, ``Q6_K``
  * **Type**: ``string``
  * **Required**: *yes*

* *output_type*

  * **Description**: the element type of the output.
  * **Range of values**: one of the types *T*
  * **Type**: ``string``
  * **Required**: *yes*


**Inputs**

* **1**: ``data`` - tensor of type ``u8`` and shape ``[..., N, B]`` with rank 2 or more: the GGUF
  blocks of ``N`` rows of ``B`` bytes, where ``B`` is a multiple of ``block_bytes``. **Required.**


**Outputs**

* **1**: ``output`` - tensor of type *T* and shape ``[..., N, K]``, ``K = B / block_bytes * 256``:
  the decoded values. ``K`` is dynamic when ``B`` is dynamic.


**Types**

* *T*: ``f32``, ``f16`` or ``bf16``.


**Example**

A *Q4_K* weight of ``4096`` rows of ``4096`` values: every row is ``16`` blocks of ``144`` bytes.

.. code-block:: xml
   :force:

   <layer ... type="GGUFDequantize" ...>
       <data format="Q4_K" output_type="f16"/>
       <input>
           <port id="0" precision="U8"> <!-- `data` -->
               <dim>4096</dim>
               <dim>2304</dim>
           </port>
       </input>
       <output>
           <port id="1" precision="FP16"> <!-- `output` -->
               <dim>4096</dim>
               <dim>4096</dim>
           </port>
       </output>
   </layer>
