// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tensor_iterator.hpp"

#include <stack>

#include "itt.hpp"

namespace ov {
op::v0::TensorIterator::TensorIterator(const OutputVector& values) : op::util::SubGraphOp(values) {}

bool op::v0::TensorIterator::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_TensorIterator_visit_attributes);
    visitor.on_attribute("body", m_bodies[0]);
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);

    return true;
}

void op::v0::TensorIterator::revalidate_and_infer_types_for_body_ops() {
    std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>> nodes_to_do;
    std::unordered_set<std::shared_ptr<Node>> nodes_done;

    for (const auto& r : m_bodies[0]->get_results()) {
        nodes_to_do.push(r);
    }
    while (nodes_to_do.size() > 0) {
        auto node = nodes_to_do.top();
        if (nodes_done.count(node) == 0) {
            OPENVINO_ASSERT(ov::as_type_ptr<op::v0::TensorIterator>(node) == nullptr, "No nested TensorIterator");
            bool can_add = true;
            size_t arg_count = node->get_input_size();
            for (size_t i = 0; i < arg_count; ++i) {
                auto dep = node->input(arg_count - i - 1).get_source_output().get_node()->shared_from_this();
                if (nodes_done.count(dep) == 0) {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            if (can_add) {
                nodes_done.insert(node);
                node->revalidate_and_infer_types();
                nodes_to_do.pop();
            }
        } else {
            nodes_to_do.pop();
        }
    }
}

void op::v0::TensorIterator::validate_and_infer_types() {
    OV_OP_SCOPE(v0_TensorIterator_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, m_bodies.size() == 1, "Number of bodies for loop is greater than 1");

    NODE_VALIDATION_CHECK(this, m_input_descriptions.size() == 1, "Loop contains input descriptions for other bodies");
    NODE_VALIDATION_CHECK(this,
                          m_output_descriptions.size() == 1,
                          "Loop contains output descriptions for other bodies");

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_input_descriptions[0].size(),
                          "Number of inputs must be the same as number of input descriptions");

    std::vector<std::shared_ptr<Node>> ends;
    m_num_iterations = -1;  // should be recalculated each shape inference from the scratch

    auto make_positive = [](int64_t value, uint64_t dim_size) -> int64_t {
        if (value < 0) {
            value = dim_size + value;
        }
        return value;
    };
    auto body = get_function();
    // Input
    for (const auto& input_description : m_input_descriptions[0]) {
        auto index = input_description->m_input_index;

        if (auto slice_input_description = ov::as_type_ptr<SliceInputDescription>(input_description)) {
            auto body_parameter = body->get_parameters().at(slice_input_description->m_body_parameter_index);
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            auto axis = slice_input_description->m_axis;
            if (input_partial_shape.rank().is_static()) {
                auto part_size = slice_input_description->m_part_size;
                // infer type for m_body_parameter
                ov::PartialShape out_shape{input_partial_shape};
                out_shape[axis] = part_size;
                body_parameter->set_partial_shape(out_shape);
                if (input_partial_shape[axis].is_static()) {
                    auto dim_size = input_partial_shape[axis].get_length();
                    auto start = make_positive(slice_input_description->m_start, dim_size);
                    auto end = make_positive(slice_input_description->m_end, dim_size);

                    // +1 because the left and right borders are included [start, end]
                    m_num_iterations = (std::abs(end - start) + 1) / part_size;
                }
            } else {
                body_parameter->set_partial_shape(ov::PartialShape::dynamic(input_partial_shape.rank()));
            }
        } else if (auto merged_input_description = ov::as_type_ptr<MergedInputDescription>(input_description)) {
            auto body_value = m_bodies[0]->get_results().at(merged_input_description->m_body_value_index)->input(0);
            ends.push_back(body_value.get_node()->shared_from_this());

            auto body_parameter = m_bodies[0]->get_parameters().at(merged_input_description->m_body_parameter_index);

            const auto& input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            body_parameter->set_partial_shape(input_partial_shape);
        } else if (auto invariant_input_description = ov::as_type_ptr<InvariantInputDescription>(input_description)) {
            auto body_parameter = m_bodies[0]->get_parameters().at(invariant_input_description->m_body_parameter_index);

            const auto& input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            body_parameter->set_partial_shape(input_partial_shape);
        }
    }

    // Body
    revalidate_and_infer_types_for_body_ops();

    // Output
    try_to_set_num_iterations_if_no_slice_inputs();

    for (const auto& output_description : m_output_descriptions[0]) {
        auto index = output_description->m_output_index;

        auto body_value = m_bodies[0]->get_results().at(output_description->m_body_value_index)->input_value(0);

        if (auto concat_output_description = ov::as_type_ptr<ConcatOutputDescription>(output_description)) {
            auto body_value_partial_shape = body_value.get_partial_shape();
            const auto& body_value_partial_rank = body_value_partial_shape.rank();
            set_output_type(index, body_value.get_element_type(), ov::PartialShape::dynamic());
            if (body_value_partial_rank.is_static()) {
                auto part_size = concat_output_description->m_part_size;
                auto axis = concat_output_description->m_axis;

                if (body_value_partial_rank == 0) {  // after scalars concatenation we must have 1D output
                    NODE_VALIDATION_CHECK(this,
                                          axis == 0,
                                          "Axis must be equal to 0 if concatenated output "
                                          "tensor slices are scalars. "
                                          "TensorIterator output index: ",
                                          index);
                    body_value_partial_shape = ov::PartialShape::dynamic(1);
                }

                body_value_partial_shape[axis] =
                    m_num_iterations != -1 ? m_num_iterations * part_size : ov::Dimension::dynamic();
                set_output_type(index, body_value.get_element_type(), body_value_partial_shape);
            }
        } else if (auto body_output_description = ov::as_type_ptr<BodyOutputDescription>(output_description)) {
            set_output_type(index, body_value.get_element_type(), body_value.get_partial_shape());
        }
    }

    NODE_VALIDATION_CHECK(this,
                          get_output_size() == m_output_descriptions[0].size(),
                          "Number of outputs must be the same as number of output descriptions");
}

namespace {
template <typename Desc>
bool has_slice_input_desc(const Desc& desc) {
    const auto is_slice_input_desc = +[](typename Desc::const_reference d) {
        return ov::is_type<op::util::SubGraphOp::SliceInputDescription>(d);
    };
    return std::any_of(begin(desc), end(desc), is_slice_input_desc);
}
}  // namespace

void op::v0::TensorIterator::try_to_set_num_iterations_if_no_slice_inputs() {
    if (m_num_iterations != -1 || has_slice_input_desc(get_input_descriptions())) {
        return;
    }

    for (const auto& output_description : m_output_descriptions[0]) {
        if (auto concat = ov::as_type_ptr<ConcatOutputDescription>(output_description)) {
            m_num_iterations = ((std::abs(concat->m_end - concat->m_start)) / concat->m_part_size);
            break;
        }
    }
}

std::shared_ptr<Node> op::v0::TensorIterator::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_TensorIterator_clone_with_new_inputs);
    auto op = std::make_shared<op::v0::TensorIterator>();
    op->set_arguments(new_args);
    op->set_output_size(m_output_descriptions.size());

    op->m_num_iterations = m_num_iterations;
    op->m_bodies[0] = get_function()->clone();

    for (auto& input_description : m_input_descriptions[0]) {
        op->m_input_descriptions[0].push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions[0]) {
        op->m_output_descriptions[0].push_back(output_description->copy());
    }
    op->validate_and_infer_types();
    return op;
}
}  // namespace ov
