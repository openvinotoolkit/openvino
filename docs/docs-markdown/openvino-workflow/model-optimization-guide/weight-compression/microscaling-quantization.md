---
sidebar_position: 2
sidebar_label: "Microscaling (MX) Quantization"
---

# Microscaling (MX) Quantization

Microscaling (MX) Quantization method has been introduced to enable users
to quantize LLMs with a high compression rate at minimal cost of accuracy.
The method helps maintain model performance comparable to that of the conventional
FP32. It increases compute and storage efficiency by using low bit-width
floating point and integer-based data formats:

<table>
  <thead>
    <tr>
      <th>Data format</th>
      <th>Data type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MXFP8</td>
      <td>FP8 (E5M2)<br/>FP8 (E4M3)</td>
      <td>Floating point, 8-bit<br/>Floating point, 8-bit</td>
    </tr>
    <tr>
      <td>MXFP6</td>
      <td>FP6 (E3M2)<br/>FP6 (E2M3)</td>
      <td>Floating point, 6-bit<br/>Floating point, 6-bit</td>
    </tr>
    <tr>
      <td><strong>MXFP4</strong></td>
      <td><strong>FP4 (E2M1)</strong></td>
      <td><strong>Floating point, 4-bit</strong></td>
    </tr>
    <tr>
      <td>MXINT8</td>
      <td>INT8</td>
      <td>Integer, 8-bit</td>
    </tr>
  </tbody>
</table>

<a id="mxfp4-support"></a>

**Currently, only the**
[MXFP4 (E2M1)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
**data format is supported in NNCF and for quantization on CPU.**
E2M1 may be considered for improving accuracy, however, quantized models will
not be faster than the ones compressed to INT8\_ASYM.

Quantization to the E2M1 data type will compress weights to 4-bit without a zero
point and with 8-bit E8M0 scales. To quantize a model to E2M1, set
`mode=CompressWeightsMode.E2M1` in `nncf.compress_weights()`. It is
recommended to use `group size = 32`. See the example below:

```python
from nncf import compress_weights, CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.E2M1, group_size=32, all_layers=True)
```

:::note
Different values for `group_size` and `ratio` are also supported.
:::

## Additional Resources

- [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Intel® Neural Compressor Documentation](https://intel.github.io/neural-compressor/latest/docs/source/3x/PT_MXQuant.html)
