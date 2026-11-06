// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note GGUFDequantize op class is under development and subject to change
///
/// \brief Decodes GGUF k-quant weight blocks, exactly as stored in a GGUF file, into a floating-point tensor.
///
/// GGUF stores every row of a quantized weight as a sequence of blocks of 256 values. Each block carries its own
/// scales, so the bytes cannot be expressed as an OpenVINO low-precision element type plus separate scale and
/// zero-point tensors without converting them. This op keeps the file bytes: a plugin with kernels for the format
/// can read them directly, any other consumer gets the decoded values through constant folding or evaluate().
///
/// Input 0: u8 tensor [..., N, B], B = K / 256 * block_bytes(format): N rows of K values each.
/// Output 0: [..., N, K] of output_type (f32, f16 or bf16).
///
/// Formats (ggml block layouts, little-endian):
///   Q4_K: 144 bytes per block: f16 d, f16 dmin, 12 bytes of 6-bit scales and mins (8 sub-blocks of 32), 128 bytes of
///         4-bit quants. w = d * sc * q - dmin * m.
///   Q5_K: 176 bytes per block: as Q4_K plus 32 bytes holding the 5th bit of every quant.
///   Q6_K: 210 bytes per block: 128 bytes low 4 bits, 64 bytes high 2 bits, 16 int8 scales (one per 16 values),
///         f16 d. w = d * sc * (q - 32).
class OPENVINO_API GGUFDequantize : public ov::op::Op {
public:
    OPENVINO_OP("GGUFDequantize");

    enum class Format { Q4_K, Q5_K, Q6_K };

    /// \brief Values per block, the same for every k-quant format.
    static constexpr size_t block_size = 256;

    GGUFDequantize() = default;
    /// \brief Constructs a GGUFDequantize operation.
    ///
    /// \param data        u8 tensor with the GGUF blocks, [..., N, K / 256 * block_bytes(format)]
    /// \param format      GGUF block format of the data
    /// \param output_type Element type of the decoded values (f32, f16 or bf16)
    GGUFDequantize(const Output<Node>& data, Format format, const ov::element::Type& output_type = ov::element::f32);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    Format get_format() const {
        return m_format;
    }
    const ov::element::Type& get_destination_type() const {
        return m_output_type;
    }

    /// \brief Bytes of one 256-value block of the format.
    static size_t block_bytes(Format format);

private:
    Format m_format = Format::Q4_K;
    ov::element::Type m_output_type = ov::element::f32;
};

}  // namespace ov::op::internal

namespace ov {
OPENVINO_API std::ostream& operator<<(std::ostream& s, const ov::op::internal::GGUFDequantize::Format& format);

template <>
class AttributeAdapter<ov::op::internal::GGUFDequantize::Format>
    : public EnumAttributeAdapterBase<ov::op::internal::GGUFDequantize::Format> {
public:
    AttributeAdapter(ov::op::internal::GGUFDequantize::Format& value)
        : EnumAttributeAdapterBase<ov::op::internal::GGUFDequantize::Format>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::internal::GGUFDequantize::Format>");
    ~AttributeAdapter() override = default;
};
}  // namespace ov
