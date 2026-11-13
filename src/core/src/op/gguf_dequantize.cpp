// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gguf_dequantize.hpp"

#include "gguf_dequantize_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/gguf_dequantize.hpp"

namespace ov {
namespace op {
namespace internal {
namespace {
reference::gguf::Format to_reference(GGUFDequantize::Format format) {
    switch (format) {
    case GGUFDequantize::Format::Q4_K:
        return reference::gguf::Format::Q4_K;
    case GGUFDequantize::Format::Q5_K:
        return reference::gguf::Format::Q5_K;
    case GGUFDequantize::Format::Q6_K:
        return reference::gguf::Format::Q6_K;
    }
    OPENVINO_THROW("Unknown GGUF block format");
}
}  // namespace

GGUFDequantize::GGUFDequantize(const Output<Node>& data, Format format, const ov::element::Type& output_type)
    : Op({data}),
      m_format(format),
      m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

size_t GGUFDequantize::block_bytes(Format format) {
    return reference::gguf::block_bytes(to_reference(format));
}

bool GGUFDequantize::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_GGUFDequantize_visit_attributes);
    visitor.on_attribute("format", m_format);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void GGUFDequantize::validate_and_infer_types() {
    OV_OP_SCOPE(internal_GGUFDequantize_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_type.is_dynamic() || data_type == ov::element::u8,
                          "The GGUF block data must be u8. Got: ",
                          data_type);
    NODE_VALIDATION_CHECK(
        this,
        m_output_type == ov::element::f32 || m_output_type == ov::element::f16 || m_output_type == ov::element::bf16,
        "The output type must be f32, f16 or bf16. Got: ",
        m_output_type);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, m_output_type, output_shapes[0]);
}

std::shared_ptr<Node> GGUFDequantize::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(internal_GGUFDequantize_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GGUFDequantize>(new_args.at(0), m_format, m_output_type);
}

bool GGUFDequantize::has_evaluate() const {
    OV_OP_SCOPE(internal_GGUFDequantize_has_evaluate);
    return get_input_element_type(0) == ov::element::u8;
}

bool GGUFDequantize::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(internal_GGUFDequantize_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1 && inputs.size() == 1);
    const auto& in = inputs[0];
    if (in.get_element_type() != ov::element::u8 || in.get_shape().size() < 2) {
        return false;
    }
    const auto bb = block_bytes(m_format);
    if (in.get_shape().back() % bb != 0) {
        return false;
    }
    const auto output_shapes = shape_infer(this, ov::util::get_tensors_partial_shapes(inputs));
    auto& out = outputs[0];
    out.set_shape(output_shapes[0].to_shape());

    const auto* src = static_cast<const uint8_t*>(in.data());
    const size_t n_blocks = in.get_byte_size() / bb;
    switch (out.get_element_type()) {
    case ov::element::f32:
        reference::gguf_dequantize(src, out.data<float>(), n_blocks, to_reference(m_format));
        return true;
    case ov::element::f16:
        reference::gguf_dequantize(src, out.data<ov::float16>(), n_blocks, to_reference(m_format));
        return true;
    case ov::element::bf16:
        reference::gguf_dequantize(src, out.data<ov::bfloat16>(), n_blocks, to_reference(m_format));
        return true;
    default:
        return false;
    }
}

}  // namespace internal
}  // namespace op

std::ostream& operator<<(std::ostream& s, const ov::op::internal::GGUFDequantize::Format& format) {
    return s << as_string(format);
}

template <>
OPENVINO_API EnumNames<ov::op::internal::GGUFDequantize::Format>&
EnumNames<ov::op::internal::GGUFDequantize::Format>::get() {
    static auto enum_names = EnumNames<ov::op::internal::GGUFDequantize::Format>(
        "ov::op::internal::GGUFDequantize::Format",
        {
            {"Q4_K", ov::op::internal::GGUFDequantize::Format::Q4_K},
            {"Q5_K", ov::op::internal::GGUFDequantize::Format::Q5_K},
            {"Q6_K", ov::op::internal::GGUFDequantize::Format::Q6_K},
        });
    return enum_names;
}
}  // namespace ov
