// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/fully_connected_gguf.hpp"

namespace ov::intel_gpu::op {

using ov::op::internal::GGUFDequantize;

FullyConnectedGGUF::FullyConnectedGGUF(const ov::Output<Node>& A, const ov::Output<Node>& B, Format format, const ov::element::Type output_type)
    : Op({A, B}),
      m_format(format),
      m_output_type(output_type) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> FullyConnectedGGUF::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<FullyConnectedGGUF>(new_args.at(0), new_args.at(1), m_format, m_output_type);
}

void FullyConnectedGGUF::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 2, "Expected 2 inputs, got ", get_input_size());
    NODE_VALIDATION_CHECK(this, get_input_element_type(1) == ov::element::u8, "GGUF weights must be u8, got ", get_input_element_type(1));
    NODE_VALIDATION_CHECK(this, is_supported(m_format), "Unsupported GGUF format ", m_format);

    const auto& a = get_input_partial_shape(0);
    const auto& w = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this, w.rank().is_static() && w.size() == 2, "GGUF weights must be 2D, got ", w);
    NODE_VALIDATION_CHECK(this, a.rank().is_dynamic() || a.size() >= 2, "Input A must have rank 2 or more, got ", a);

    if (w[1].is_static() && a.rank().is_static() && a[a.size() - 1].is_static()) {
        const auto block_bytes = static_cast<int64_t>(GGUFDequantize::block_bytes(m_format));
        const auto k = w[1].get_length() / block_bytes * static_cast<int64_t>(GGUFDequantize::block_size);
        NODE_VALIDATION_CHECK(this,
                              w[1].get_length() % block_bytes == 0 && k == a[a.size() - 1].get_length(),
                              "Input K (",
                              a[a.size() - 1],
                              ") does not match the ",
                              m_format,
                              " weights (",
                              w[1],
                              " bytes per row)");
    }

    ov::PartialShape out = a;
    if (out.rank().is_static()) {
        out[out.size() - 1] = w[0];
    }
    const auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out);
}

bool FullyConnectedGGUF::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("format", m_format);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

}  // namespace ov::intel_gpu::op
