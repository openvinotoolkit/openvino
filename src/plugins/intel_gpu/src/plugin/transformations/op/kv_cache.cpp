// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "gather_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& beam_idx,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data, beam_idx})
    , m_concat_axis(concat_axis)
    , m_gather_axis(gather_axis)
    , m_indirect(true)
    , m_output_type(output_type) {
    m_variable = past_variable;
    if (m_indirect)
        set_output_size(2);
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data})
    , m_concat_axis(concat_axis)
    , m_gather_axis(0)
    , m_indirect(false)
    , m_output_type(output_type) {
    m_variable = past_variable;
    validate_and_infer_types();
}

bool KVCache::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("concat_axis", m_concat_axis);
    visitor.on_attribute("gather_axis", m_gather_axis);
    visitor.on_attribute("indirect", m_indirect);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void KVCache::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    std::vector<ov::PartialShape> input_shapes = {m_variable->get_info().data_shape, get_input_partial_shape(1)};
    if (get_output_size() == 2)
        input_shapes.push_back(get_input_partial_shape(2));
    auto shapes = shape_infer(this, input_shapes);
    set_output_type(0, output_type, shapes[0]);
    if (m_indirect) {
        set_output_type(1, get_input_element_type(2), shapes[1]);
    }
}

std::shared_ptr<Node> KVCache::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         m_variable,
                                         m_concat_axis,
                                         m_output_type);

    } else {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         m_variable,
                                         m_concat_axis,
                                         m_gather_axis,
                                         m_output_type);
    }
}

std::vector<ov::PartialShape> shape_infer(const KVCache* op, std::vector<ov::PartialShape> input_shapes) {
    ov::op::v0::Concat concat;
    concat.set_axis(op->get_concat_axis());
    std::vector<ov::PartialShape> out_shapes;

    if (op->get_output_size() == 2) {
        ov::op::v8::Gather gather;
        int64_t gather_axis = ov::util::normalize(op->get_gather_axis(), input_shapes[0].size());
        auto gather_axis_tensor = ov::Tensor(ov::element::i64, ov::Shape{1}, static_cast<void*>(&gather_axis));
        std::unordered_map<size_t, ov::Tensor> gather_axis_data = {{2, gather_axis_tensor}};
        std::vector<ov::PartialShape> gather_inputs = {input_shapes[0], input_shapes[2], ov::PartialShape{1}};
        auto gather_out_shapes = ov::op::shape_infer(&gather, gather_inputs, ov::make_tensor_accessor(gather_axis_data));
        std::vector<ov::PartialShape> concat_shapes = {gather_out_shapes[0], input_shapes[1]};
        out_shapes = ov::op::v0::shape_infer(&concat, concat_shapes);
        int64_t concat_axis = ov::util::normalize(op->get_concat_axis(), input_shapes[0].size());
        ov::PartialShape beam_table_shape(std::vector<size_t>(out_shapes[0].size(), 1));
        beam_table_shape[gather_axis] = out_shapes[0][gather_axis];
        beam_table_shape[concat_axis] = out_shapes[0][concat_axis];
        out_shapes.push_back(beam_table_shape);
    } else {
        std::vector<ov::PartialShape> concat_shapes = {input_shapes[0], input_shapes[1]};
        out_shapes = ov::op::v0::shape_infer(&concat, concat_shapes);
    }

    return out_shapes;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
