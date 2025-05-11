// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/loop.hpp"

#include <climits>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/reference/loop.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace op {
namespace v5 {
Loop::Loop(const Output<Node>& trip_count, const Output<Node>& execution_condition) : SubGraphOp() {
    set_argument(0, trip_count);
    set_argument(1, execution_condition);
}

bool Loop::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_Loop_visit_attributes);
    visitor.on_attribute("body", m_bodies[0]);
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);
    visitor.on_attribute("special_body_ports", m_special_body_ports);

    return true;
}

void Loop::validate_and_infer_types() {
    OV_OP_SCOPE(v5_Loop_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, m_bodies.size() == 1, "Number of bodies for loop is greater than 1");

    NODE_VALIDATION_CHECK(this, m_input_descriptions.size() == 1, "Loop contains input descriptions for other bodies");
    NODE_VALIDATION_CHECK(this,
                          m_output_descriptions.size() == 1,
                          "Loop contains output descriptions for other bodies");

    if (m_special_body_ports.current_iteration_input_idx >= 0) {
        const auto& cur_iter_rank = m_bodies[0]
                                        ->get_parameters()
                                        .at(m_special_body_ports.current_iteration_input_idx)
                                        ->get_partial_shape()
                                        .rank();
        NODE_VALIDATION_CHECK(this,
                              ov::util::is_rank_compatible_any_of(cur_iter_rank, {0, 1}),
                              "Rank of CurrentIteration input must be equal to 0 or 1");
    }
    bool zero_number_of_iter = false;
    const auto& loop_execution_condition = input_value(1);
    const auto& loop_condition_rank = loop_execution_condition.get_partial_shape().rank();
    NODE_VALIDATION_CHECK(this,
                          ov::util::is_rank_compatible_any_of(loop_condition_rank, {0, 1}),
                          "Rank of ExecutionCondition input must be equal to 0 or 1");
    if (const auto& cond_value = ov::util::get_constant_from_source(loop_execution_condition)) {
        auto val = cond_value->cast_vector<bool>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the Condition constant is greater than 1");

        if (!val[0]) {
            zero_number_of_iter = true;
        }
    }

    bool condition_always_true = false;
    if (m_special_body_ports.body_condition_output_idx < 0)
        // special body ports were not set yet, so we can't calculate output shape
        return;

    const auto& body_execution_condition =
        m_bodies[0]->get_results().at(m_special_body_ports.body_condition_output_idx)->input_value(0);
    const auto& body_condition_rank = body_execution_condition.get_partial_shape().rank();
    NODE_VALIDATION_CHECK(this,
                          ov::util::is_rank_compatible_any_of(body_condition_rank, {0, 1}),
                          "Rank of BodyExecutionCondition output must be equal to 0 or 1");
    if (const auto& cond_value = ov::util::get_constant_from_source(body_execution_condition)) {
        auto val = cond_value->cast_vector<bool>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the Condition constant is greater than 1");

        if (val[0]) {
            condition_always_true = true;
        } else {
            m_num_iterations = 1;  // condition_always_false, do_while mode
        }
    } else if (const auto& cond_param =
                   ov::as_type_ptr<const op::v0::Parameter>(body_execution_condition.get_node_shared_ptr())) {
        // Const(true or false) -> Loop (body: Parameter -> execution_condition output)
        for (const auto& desc : get_input_descriptions()) {
            if (m_bodies[0]->get_parameters().at(desc->m_body_parameter_index) == cond_param) {
                if (const auto& cond_value = ov::util::get_constant_from_source(input_value(desc->m_input_index))) {
                    auto val = cond_value->cast_vector<bool>();
                    NODE_VALIDATION_CHECK(this,
                                          val.size() == 1,
                                          "The number of values in the Condition constant is greater than 1");

                    if (val[0]) {
                        condition_always_true = true;
                    } else {
                        m_num_iterations = 1;  // condition_always_false, do_while mode
                    }
                }
            }
        }
    }

    const auto& trip_count = input_value(0);
    const auto& trip_count_rank = trip_count.get_partial_shape().rank();
    NODE_VALIDATION_CHECK(this,
                          ov::util::is_rank_compatible_any_of(trip_count_rank, {0, 1}),
                          "Rank of TripCount input must be equal to 0 or 1");
    if (const auto& trip_count_val = ov::util::get_constant_from_source(trip_count)) {
        auto val = trip_count_val->cast_vector<int64_t>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the TripCount constant is greater than 1");
        if (condition_always_true)
            m_num_iterations = val[0];
    }

    // WA: input description with index 0 or 1 means that Loop consructor will duplicate it in
    // the inputs.
    // When using visit_attributes() no duplication occurs, input_offset shall be decremented.
    int64_t input_offset = 2;
    for (const auto& in_desc : m_input_descriptions[0]) {
        if (in_desc->m_input_index == 0 || in_desc->m_input_index == 1) {
            input_offset--;
        }
    }
    // input_offset < 0 means that there are several duplications of external_port_id
    // (the same ext_port_id is connected to several Parameters in the port map) in input_desc,
    // this can lead to wrong or undefined behavior, so throw exception here. Ticket: 47302
    NODE_VALIDATION_CHECK(this, input_offset >= 0, "External port id 0 or 1 is duplicated.");

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_input_descriptions[0].size() + input_offset,
                          "Number of inputs must be the same as number of input descriptions");

    // if output shape is not same with input in a back-edge, here will update the input shape
    // Port map processing: output -> input
    std::map<uint64_t, uint64_t> back_edges;

    // Input
    for (const auto& input_description : m_input_descriptions[0]) {
        auto index = input_description->m_input_index;

        if (auto slice_input_description = as_type_ptr<SliceInputDescription>(input_description)) {
            auto body_parameter = m_bodies[0]->get_parameters().at(slice_input_description->m_body_parameter_index);
            const auto& input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            const auto& input_type = inputs().at(index).get_source_output().get_element_type();
            body_parameter->set_element_type(input_type);
            if (input_partial_shape.rank().is_dynamic()) {
                body_parameter->set_partial_shape(PartialShape::dynamic());
            } else {
                auto out_shape = input_partial_shape;
                const auto axis =
                    ov::util::try_normalize_axis(slice_input_description->m_axis, input_partial_shape.rank(), *this);
                out_shape[axis] = slice_input_description->m_part_size;
                body_parameter->set_partial_shape(out_shape);
            }
        } else if (auto merged_input_description = as_type_ptr<MergedInputDescription>(input_description)) {
            auto body_value = m_bodies[0]->get_results().at(merged_input_description->m_body_value_index);

            auto body_parameter = m_bodies[0]->get_parameters().at(merged_input_description->m_body_parameter_index);

            const auto& input_partial_shape = input(index).get_partial_shape();
            const auto& input_type = input(index).get_element_type();

            body_parameter->set_partial_shape(input_partial_shape);
            body_parameter->set_element_type(input_type);
            back_edges[merged_input_description->m_body_value_index] = merged_input_description->m_body_parameter_index;
        } else if (auto invariant_input_description =
                       as_type_ptr<op::v0::TensorIterator::InvariantInputDescription>(input_description)) {
            auto body_parameter = m_bodies[0]->get_parameters().at(invariant_input_description->m_body_parameter_index);

            const auto& input_partial_shape = input(index).get_partial_shape();
            const auto& input_type = input(index).get_element_type();

            body_parameter->set_partial_shape(input_partial_shape);
            body_parameter->set_element_type(input_type);
        }
    }

    // Body
    m_bodies[0]->validate_nodes_and_infer_types();

    if (!back_edges.empty()) {
        // if an exact value is available, limit the number of iterations.
        size_t i, max_num_of_iterations = m_num_iterations == -1 ? INT_MAX : m_num_iterations;
        bool need_reinvalidate = false;
        for (i = 0; i < max_num_of_iterations; i++) {
            need_reinvalidate = false;
            for (const auto& output_description : m_output_descriptions[0]) {
                auto body_value = m_bodies[0]->get_results().at(output_description->m_body_value_index)->input_value(0);

                if (auto body_output_description =
                        as_type_ptr<v0::TensorIterator::BodyOutputDescription>(output_description)) {
                    if (!back_edges.count(output_description->m_body_value_index))
                        continue;
                    const auto& body_value_shape = body_value.get_partial_shape();
                    auto input_param =
                        m_bodies[0]->get_parameters().at(back_edges[output_description->m_body_value_index]);
                    const auto& input_param_ps = input_param->get_partial_shape();
                    if (body_value_shape.rank().is_static()) {
                        // handle the case: when sub-model's output shape does not compatible input shape in a
                        // back-edge, such as
                        //          Parameter(out:-1, 1)->|
                        //                                |->Concat(out:-1, 2)->Result(out:-1, 2)
                        //          Parameter(out:-1, 1)->|   (axis==1)
                        // when iteration number is unknown or sub-model output shape may be vary, the Result shape
                        // should infer as (-1, -1), then set changed one to input and propagate to others.
                        if (input_param_ps.rank().is_static()) {
                            const auto body_rank_len = body_value_shape.rank().get_length();
                            const auto input_rank_len = input_param_ps.rank().get_length();
                            PartialShape new_ps;
                            bool shape_changed = false;
                            if (body_rank_len == input_rank_len) {
                                new_ps = body_value_shape;
                                for (auto j = 0; j < body_rank_len; j++) {
                                    if (!body_value_shape[j].compatible(input_param_ps[j])) {
                                        new_ps[j] = Dimension::dynamic();
                                        shape_changed = true;
                                    }
                                }
                            } else {
                                new_ps = PartialShape::dynamic();
                                shape_changed = true;
                            }
                            // reset sub model input shape
                            if (shape_changed) {
                                need_reinvalidate = true;
                                input_param->set_partial_shape(new_ps);
                            }
                        }
                    } else {
                        if (input_param_ps.rank().is_static()) {
                            // output shape is dynamic, let the input known now we are dynamic shape
                            input_param->set_partial_shape(body_value_shape);
                            need_reinvalidate = true;
                        }
                    }
                }
            }
            // only input shape changed we will re-compute output shape
            if (need_reinvalidate) {
                m_bodies[0]->validate_nodes_and_infer_types();
            } else {
                break;
            }
        }
    }

    // Output
    for (const auto& output_description : m_output_descriptions[0]) {
        auto index = output_description->m_output_index;

        auto body_value = m_bodies[0]->get_results().at(output_description->m_body_value_index)->input_value(0);

        if (auto concat_output_description =
                as_type_ptr<v0::TensorIterator::ConcatOutputDescription>(output_description)) {
            const auto& body_value_partial_shape = body_value.get_partial_shape();
            auto out_shape = body_value_partial_shape;
            if (zero_number_of_iter) {
                out_shape = PartialShape{0};
            } else if (out_shape.rank().is_static()) {
                const auto axis =
                    ov::util::try_normalize_axis(concat_output_description->m_axis, out_shape.rank(), *this);
                const auto rank = out_shape.rank().get_length();
                if (rank == 0) {
                    out_shape = PartialShape{1};
                }

                if (out_shape[axis].is_static() && m_num_iterations != -1) {
                    out_shape[axis] = Dimension{out_shape[axis].get_length() * m_num_iterations};
                } else {
                    out_shape[axis] = Dimension::dynamic();
                }
            }
            set_output_type(index, body_value.get_element_type(), out_shape);
        }

        else if (auto body_output_description =
                     as_type_ptr<v0::TensorIterator::BodyOutputDescription>(output_description)) {
            const auto& body_value_shape = body_value.get_partial_shape();
            if (body_value_shape.is_dynamic()) {
                set_output_type(index, body_value.get_element_type(), body_value_shape);
            } else {
                auto shape = body_value_shape.get_shape();
                if (zero_number_of_iter) {
                    shape.at(0) = 0;
                }
                set_output_type(index, body_value.get_element_type(), shape);
            }
        }
    }

    NODE_VALIDATION_CHECK(this,
                          get_output_size() == m_output_descriptions[0].size(),
                          "Number of outputs must be the same as number of output descriptions");
}

std::shared_ptr<Node> Loop::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_Loop_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto op = std::make_shared<Loop>();
    OPENVINO_ASSERT(op.get(),
                    op != nullptr,
                    "Cannot clone ",
                    description(),
                    " operation with name ",
                    get_friendly_name());
    clone_to(*op, new_args);
    return op;
}

Output<Node> Loop::get_concatenated_slices(const Output<Node>& value,
                                           int64_t start,
                                           int64_t stride,
                                           int64_t part_size,
                                           int64_t end,
                                           int64_t axis) {
    OPENVINO_ASSERT(start == 0 && stride == 1 && part_size == 1 && end == -1,
                    "Invalid start, stride, part_size, or end attribute values in Loop op. "
                    "Supported values for start {0}, for stride and part_size {1}, for end "
                    "{-1}");
    return SubGraphOp::get_concatenated_slices(value, start, stride, part_size, end, axis);
}

bool Loop::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v5_Loop_evaluate);
    EvaluationContext evaluation_context;
    reference::loop(m_bodies[0],
                    m_output_descriptions[0],
                    m_input_descriptions[0],
                    m_special_body_ports,
                    outputs,
                    inputs,
                    evaluation_context);
    return true;
}

bool Loop::evaluate(TensorVector& outputs,
                    const TensorVector& inputs,
                    const EvaluationContext& evaluation_context) const {
    OV_OP_SCOPE(v5_Loop_evaluate);
    reference::loop(m_bodies[0],
                    m_output_descriptions[0],
                    m_input_descriptions[0],
                    m_special_body_ports,
                    outputs,
                    inputs,
                    evaluation_context);
    return true;
}

bool Loop::has_evaluate() const {
    OV_OP_SCOPE(v5_Loop_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
    case element::i64:
        return true;
    default:
        return false;
    }
}

void Loop::clone_to(Loop& dst, const OutputVector& new_args) const {
    dst.set_arguments(new_args);
    dst.set_output_size(m_output_descriptions.size());

    dst.m_num_iterations = m_num_iterations;
    dst.m_special_body_ports = m_special_body_ports;

    dst.m_bodies[0] = get_function()->clone();

    for (auto& input_description : m_input_descriptions[0]) {
        dst.m_input_descriptions[0].push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions[0]) {
        dst.m_output_descriptions[0].push_back(output_description->copy());
    }
    dst.validate_and_infer_types();
}

Loop::Loop(const op::v5::Loop& other) : SubGraphOp() {
    other.clone_to(*this, other.input_values());
}
}  // namespace v5
}  // namespace op

AttributeAdapter<op::v5::Loop::SpecialBodyPorts>::~AttributeAdapter() = default;
}  // namespace ov
