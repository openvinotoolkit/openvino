// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/swiglu.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/variadic_split.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

SwiGLU::SwiGLU(const Output<Node>& data,
               int64_t axis,
               int64_t split_lengths,
               const GluType glu_type,
               const size_t split_to_glu_idx,
               const ov::element::Type output_type)
    : Op({data}), m_axis(axis), m_split_lengths(split_lengths),
      m_glu_type(glu_type), m_split_to_glu_idx(split_to_glu_idx), m_output_type(output_type) {
    validate_and_infer_types();
}

bool SwiGLU::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("split_lengths", m_split_lengths);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void SwiGLU::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;

    std::vector<ov::PartialShape> input_shapes = {
        get_input_partial_shape(0),
        ov::PartialShape(ov::Shape{}),
        ov::PartialShape(ov::Shape{2})
    };

    set_output_type(0, output_type, shape_infer(this, input_shapes)[0]);
}

std::shared_ptr<Node> SwiGLU::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SwiGLU>(new_args.at(0),
                                    m_axis,
                                    m_split_lengths,
                                    m_glu_type,
                                    m_split_to_glu_idx,
                                    m_output_type);
}

std::vector<ov::PartialShape> shape_infer(const SwiGLU* op, std::vector<ov::PartialShape> input_shapes) {
    ov::op::v1::VariadicSplit variadic_split;
    std::vector<int64_t> axis = { op->get_axis() };
    std::vector<int64_t> split_lengths = { op->get_split_lengths(), -1 };

    std::unordered_map<size_t, ov::Tensor> const_data;
    const_data.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{}, static_cast<void*>(axis.data())));
    const_data.emplace(2, ov::Tensor(ov::element::i64, ov::Shape{split_lengths.size()}, static_cast<void*>(split_lengths.data())));

    return ov::op::v1::shape_infer(&variadic_split, input_shapes, ov::make_tensor_accessor(const_data));
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
