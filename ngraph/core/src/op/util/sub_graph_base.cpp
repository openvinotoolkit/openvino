// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/sub_graph_base.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset5.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::SubGraphOp, "SubGraphOp", 0);

op::util::SubGraphOp::SubGraphOp() : MultiSubGraphOp(1) {}

op::util::SubGraphOp::SubGraphOp(const OutputVector& args) : MultiSubGraphOp(args, 1) {}

void op::util::SubGraphOp::set_merged_input(const std::shared_ptr<Parameter>& body_parameter,
                                            const Output<Node>& initial_value,
                                            const Output<Node>& successive_value) {
    auto body = get_function();

    m_input_descriptions[0].push_back(
        std::make_shared<TensorIterator::MergedInputDescription>(input_for_value(initial_value).get_index(),
                                                                 body->get_parameter_index(body_parameter),
                                                                 body->get_result_index(successive_value)));
    validate_and_infer_types();
}

void op::util::SubGraphOp::set_invariant_input(const std::shared_ptr<Parameter>& body_parameter,
                                               const Output<Node>& value) {
    auto body = get_function();
    m_input_descriptions[0].push_back(
        std::make_shared<TensorIterator::InvariantInputDescription>(input_for_value(value).get_index(),
                                                                    body->get_parameter_index(body_parameter)));
    validate_and_infer_types();
}

Output<Node> op::util::SubGraphOp::get_iter_value(const Output<Node>& body_value, int64_t iteration) {
    auto output_index = get_output_size();
    auto body = get_function();
    m_output_descriptions[0].push_back(
        std::make_shared<BodyOutputDescription>(body->get_result_index(body_value), output_index, iteration));
    set_output_size(output_index + 1);
    validate_and_infer_types();
    return Output<Node>(shared_from_this(), output_index);
}

Output<Node> op::util::SubGraphOp::get_concatenated_slices(const Output<Node>& body_value,
                                                           int64_t start,
                                                           int64_t stride,
                                                           int64_t part_size,
                                                           int64_t end,
                                                           int64_t axis) {
    auto output_index = get_output_size();
    auto body = get_function();
    m_output_descriptions[0].push_back(std::make_shared<ConcatOutputDescription>(body->get_result_index(body_value),
                                                                                 output_index,
                                                                                 start,
                                                                                 stride,
                                                                                 part_size,
                                                                                 end,
                                                                                 axis));
    set_output_size(output_index + 1);
    validate_and_infer_types();
    return Output<Node>(shared_from_this(), output_index);
}

void op::util::SubGraphOp::set_sliced_input(const std::shared_ptr<Parameter>& parameter,
                                            const Output<Node>& value,
                                            int64_t start,
                                            int64_t stride,
                                            int64_t part_size,
                                            int64_t end,
                                            int64_t axis) {
    auto body = get_function();
    m_input_descriptions[0].push_back(std::make_shared<SliceInputDescription>(input_for_value(value).get_index(),
                                                                              body->get_parameter_index(parameter),
                                                                              start,
                                                                              stride,
                                                                              part_size,
                                                                              end,
                                                                              axis));
    validate_and_infer_types();
}

Input<Node> op::util::SubGraphOp::input_for_value(const Output<Node>& value) {
    auto input_index = get_input_size();
    set_argument(input_index, value);
    return Input<Node>(this, input_index);
}
