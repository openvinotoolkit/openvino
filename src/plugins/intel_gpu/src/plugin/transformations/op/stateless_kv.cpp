// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/stateless_kv.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov::intel_gpu::op {

StatelessKV::StatelessKV(const OutputVector& inputs, int64_t concat_axis, bool is_present_len)
    : Op(inputs),
      m_concat_axis(concat_axis),
      m_is_present_len(is_present_len) {}

StatelessKV::StatelessKV(const Output<Node>& past,
                         const Output<Node>& new_token_data,
                         const Output<Node>& present_seq_len,
                         int64_t concat_axis,
                         bool is_present_len)
    : StatelessKV({past, new_token_data, present_seq_len}, concat_axis, is_present_len) {
    validate_and_infer_types();
}

StatelessKV::StatelessKV(const Output<Node>& past,
                         const Output<Node>& new_token_data,
                         const Output<Node>& present_seq_len,
                         const Output<Node>& pos_idx,
                         int64_t concat_axis,
                         bool is_present_len)
    : StatelessKV({past, new_token_data, present_seq_len, pos_idx}, concat_axis, is_present_len) {
    validate_and_infer_types();
}

bool StatelessKV::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("concat_axis", m_concat_axis);
    visitor.on_attribute("is_present_len", m_is_present_len);
    return true;
}

void StatelessKV::validate_and_infer_types() {
    const auto input_type = get_input_element_type(0);
    const auto& input_shape = get_input_partial_shape(0);
    const auto& append_shape = get_input_partial_shape(1);

    OPENVINO_ASSERT(input_shape.rank().is_static() && append_shape.rank().is_static(), "[GPU] stateless_kv requires static input rank");
    OPENVINO_ASSERT(input_shape.rank() == append_shape.rank(), "[GPU] stateless_kv requires input and new_token being the same rank");
    const auto concat_axis = ov::util::normalize(m_concat_axis, append_shape.rank().get_length());
    OPENVINO_ASSERT(concat_axis >= 0 && static_cast<size_t>(concat_axis) < static_cast<size_t>(input_shape.rank().get_length()),
                    "[GPU] stateless_kv concat_axis exceeds input rank");
    m_concat_axis = concat_axis;  // pre-compute normalized axis for later use

    std::vector<ov::PartialShape> input_shapes = {input_shape, append_shape};

    auto shapes = shape_infer(this, input_shapes);

    set_output_type(0, input_type, shapes[0]);
    set_output_type(1, input_type, shapes[1]);
}

std::shared_ptr<Node> StatelessKV::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<StatelessKV>(new_args.at(0), new_args.at(1), new_args.at(2), m_concat_axis, m_is_present_len);
    }
    return std::make_shared<StatelessKV>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_concat_axis, m_is_present_len);
}

std::vector<ov::PartialShape> shape_infer(const StatelessKV* op, const std::vector<ov::PartialShape>& input_shapes) {
    const auto concat_axis = op->get_concat_axis();
    OPENVINO_ASSERT(concat_axis >= 0);
    std::vector<ov::PartialShape> out_shapes(2, input_shapes[0]);
    auto& full_shape = out_shapes[0];
    auto& trim_shape = out_shapes[1];
    trim_shape[concat_axis] = Dimension{};
    const auto update_offset = op->get_update_offset();

    if (update_offset && input_shapes[0][concat_axis].is_static() && input_shapes[1][concat_axis].is_static()) {
        const auto updated_dim = input_shapes[1][concat_axis] + *update_offset;
        // OPENVINO_ASSERT(updated_dim.get_length() <= full_shape[concat_axis].get_length());
        trim_shape[concat_axis] = updated_dim;
        if (updated_dim.get_length() > full_shape[concat_axis].get_length()) {
            full_shape[concat_axis] = updated_dim;
        }
    }

    return out_shapes;
}

}  // namespace ov::intel_gpu::op
