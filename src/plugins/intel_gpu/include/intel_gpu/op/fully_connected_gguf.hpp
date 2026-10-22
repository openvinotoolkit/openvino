// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/gguf_dequantize.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// Fully connected layer whose weights are GGUF k-quant blocks, read by the kernels without conversion.
///   A      : [..., M, K], f16 or f32
///   B      : u8 constant [N, K / 256 * block_bytes], the blocks of N rows
///            Q5_K: the 176-byte blocks as stored.
///            Q6_K: the 210-byte blocks are only 2-byte aligned, so their bytes are regrouped into planes over all
///                  N * K / 256 blocks: QL [nblk][128] | QH [nblk][64] | SC [nblk][16] | D [nblk] (f16).
///   output : [..., M, N]
class FullyConnectedGGUF : public ov::op::Op {
public:
    OPENVINO_OP("FullyConnectedGGUF", "gpu_opset");

    using Format = ov::op::internal::GGUFDequantize::Format;

    FullyConnectedGGUF() = default;

    FullyConnectedGGUF(const ov::Output<Node>& A, const ov::Output<Node>& B, Format format, const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    Format get_format() const {
        return m_format;
    }
    ov::element::Type get_output_type() const {
        return m_output_type;
    }

    static bool is_supported(Format format) {
        return format == Format::Q5_K || format == Format::Q6_K;
    }

protected:
    Format m_format = Format::Q6_K;
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::intel_gpu::op
