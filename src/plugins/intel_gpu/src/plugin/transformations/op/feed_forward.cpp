// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/feed_forward.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

FeedForward::FeedForward(const Output<Node>& data,
                const Output<Node>& c1,
                const Output<Node>& c2,
                const Output<Node>& c3,
                const Output<Node>& c4,
                const ov::element::Type output_type)
    : Op({data, c1, c2, c3, c4}), m_output_type(output_type) {
    validate_and_infer_types();
}

bool FeedForward::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void FeedForward::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    set_output_type(0, output_type, input_shapes[0]);
}

std::shared_ptr<Node> FeedForward::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<FeedForward>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    new_args.at(4),
                                    m_output_type);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
