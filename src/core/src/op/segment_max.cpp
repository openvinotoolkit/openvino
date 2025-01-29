// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "segment_max_shape_inference.hpp"

namespace ov {
namespace op {
namespace v16 {

SegmentMax::SegmentMax(const Output<Node>& data, const Output<Node>& segment_ids, const op::FillMode fill_mode)
    : Op({data, segment_ids}),
      m_fill_mode(fill_mode) {
    constructor_validate_and_infer_types();
}

SegmentMax::SegmentMax(const Output<Node>& data,
                       const Output<Node>& segment_ids,
                       const Output<Node>& num_segments,
                       const op::FillMode fill_mode)
    : Op({data, segment_ids, num_segments}),
      m_fill_mode(fill_mode) {
    constructor_validate_and_infer_types();
}

bool SegmentMax::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v16_SegmentMax_visit_attributes);
    visitor.on_attribute("fill_mode", m_fill_mode);
    return true;
}

void SegmentMax::validate_and_infer_types() {
    OV_OP_SCOPE(v16_SegmentMax_validate_and_infer_types);
    const auto& segment_ids_element_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          segment_ids_element_type == element::i32 || segment_ids_element_type == element::i64,
                          "The element type of the segment_ids input be i32 or i64. Got: ",
                          segment_ids_element_type);
    if (inputs().size() == 3) {
        const auto& num_segments_element_type = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              num_segments_element_type == element::i32 || num_segments_element_type == element::i64,
                              "The element type of the num_segments input be i32 or i64. Got: ",
                              num_segments_element_type);
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> SegmentMax::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v16_SegmentMax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<SegmentMax>(new_args.at(0), new_args.at(1), new_args.at(2), m_fill_mode);
    } else {
        return std::make_shared<SegmentMax>(new_args.at(0), new_args.at(1), m_fill_mode);
    }
}

const op::FillMode SegmentMax::get_fill_mode() const {
    return m_fill_mode;
}

}  // namespace v16
}  // namespace op
}  // namespace ov
