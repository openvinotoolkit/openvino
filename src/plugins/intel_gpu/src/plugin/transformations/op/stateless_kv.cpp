// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/stateless_kv.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

#include <algorithm>

namespace ov::intel_gpu::op {

StatelessKV::StatelessKV(const OutputVector& inputs, int64_t concat_axis, bool is_seq_len_present_len, int64_t window_size)
    : Op(inputs),
      m_concat_axis(concat_axis),
      m_is_seq_len_present_len(is_seq_len_present_len),
      m_window_size(window_size) {}

StatelessKV::StatelessKV(const Output<Node>& past,
                         const Output<Node>& new_token_data,
                         const Output<Node>& seq_len,
                         int64_t concat_axis,
                         bool is_seq_len_present_len,
                         int64_t window_size)
    : StatelessKV({past, new_token_data, seq_len}, concat_axis, is_seq_len_present_len, window_size) {
    validate_and_infer_types();
}

StatelessKV::StatelessKV(const Output<Node>& past,
                         const Output<Node>& new_token_data,
                         const Output<Node>& seq_len,
                         const Output<Node>& pos_idx,
                         int64_t concat_axis,
                         bool is_seq_len_present_len)
    : StatelessKV({past, new_token_data, seq_len, pos_idx}, concat_axis, is_seq_len_present_len, 0) {
    validate_and_infer_types();
}

bool StatelessKV::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("concat_axis", m_concat_axis);
    visitor.on_attribute("is_seq_len_present_len", m_is_seq_len_present_len);
    visitor.on_attribute("window_size", m_window_size);
    return true;
}

void StatelessKV::validate_and_infer_types() {
    const auto input_type = get_input_element_type(0);
    const auto& input_shape = get_input_partial_shape(0);
    const auto& append_shape = get_input_partial_shape(1);
    const auto& seq_len_shape = get_input_partial_shape(2);
    const auto& seq_len_dtype = get_input_element_type(2);

    OPENVINO_ASSERT(m_window_size >= 0, "[GPU] stateless_kv window_size must be non-negative");
    OPENVINO_ASSERT(m_window_size == 0 || get_input_size() == 3,
                    "[GPU] stateless_kv doesn't support window_size together with pos_idx");
    OPENVINO_ASSERT(input_shape.rank().is_static() && append_shape.rank().is_static(), "[GPU] stateless_kv requires static input rank");
    OPENVINO_ASSERT(input_shape.rank() == append_shape.rank(), "[GPU] stateless_kv requires input and new_token being the same rank");
    OPENVINO_ASSERT(seq_len_dtype.is_integral_number() && seq_len_dtype != element::Type_t::u64, "[GPU] stateless_kv requires seq_len to be an integer number");
    OPENVINO_ASSERT(seq_len_shape.is_static() && shape_size(seq_len_shape.get_shape()) == 1,
                    "[GPU] stateless_kv requires seq_len to be static shape with single element");
    const auto concat_axis = ov::util::normalize(m_concat_axis, append_shape.rank().get_length());
    OPENVINO_ASSERT(concat_axis >= 0 && static_cast<size_t>(concat_axis) < static_cast<size_t>(input_shape.rank().get_length()),
                    "[GPU] stateless_kv concat_axis exceeds input rank");
    OPENVINO_ASSERT(input_shape[concat_axis].is_dynamic() || m_window_size <= input_shape[concat_axis].get_length(),
                    "[GPU] stateless_kv window_size exceeds cache capacity");
    m_concat_axis = concat_axis;  // pre-compute normalized axis for later use

    std::vector<ov::PartialShape> input_shapes = {input_shape, append_shape};

    auto shapes = shape_infer(this, input_shapes);

    set_output_type(0, input_type, shapes[0]);
    set_output_type(1, input_type, shapes[1]);
}

std::shared_ptr<Node> StatelessKV::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<StatelessKV>(new_args.at(0), new_args.at(1), new_args.at(2), m_concat_axis, m_is_seq_len_present_len, m_window_size);
    }
    return std::make_shared<StatelessKV>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_concat_axis, m_is_seq_len_present_len);
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
        const auto updated_len = input_shapes[1][concat_axis].get_length() + *update_offset;
        OPENVINO_ASSERT(updated_len >= 0, "[GPU] stateless_kv update offset produces a negative output length");
        if (op->get_window_size() > 0) {
            const auto windowed_len = input_shapes[1][concat_axis].get_length() + op->get_window_size() - 1;
            if (op->get_is_rolling()) {
                OPENVINO_ASSERT(op->get_past_seq_len().has_value(), "[GPU] stateless_kv rolling SWA shape inference requires past_seq_len");
                const auto present_len = *op->get_past_seq_len() + input_shapes[1][concat_axis].get_length();
                trim_shape[concat_axis] = std::min(windowed_len, present_len);
            } else {
                trim_shape[concat_axis] = std::min(windowed_len, updated_len);
            }
        } else {
            const Dimension updated_dim{updated_len};
            trim_shape[concat_axis] = updated_dim;
            if (updated_dim.get_length() > full_shape[concat_axis].get_length()) {
                full_shape[concat_axis] = updated_dim;
            }
        }
    }

    return out_shapes;
}

}  // namespace ov::intel_gpu::op
