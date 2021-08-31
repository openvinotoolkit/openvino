// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/multi_subgraph_base.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset5.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::MultiSubGraphOp, "MultiSubGraphOp", 0);
NGRAPH_RTTI_DEFINITION(op::util::MultiSubGraphOp::SliceInputDescription, "SliceInputDescription", 0);
NGRAPH_RTTI_DEFINITION(op::util::MultiSubGraphOp::MergedInputDescription, "MergedInputDescription", 0);
NGRAPH_RTTI_DEFINITION(op::util::MultiSubGraphOp::InvariantInputDescription, "InvariantInputDescription", 0);
NGRAPH_RTTI_DEFINITION(op::util::MultiSubGraphOp::BodyOutputDescription, "BodyOutputDescription", 0);
NGRAPH_RTTI_DEFINITION(op::util::MultiSubGraphOp::ConcatOutputDescription, "ConcatOutputDescription", 0);

op::util::MultiSubGraphOp::InputDescription::InputDescription(uint64_t input_index, uint64_t body_parameter_index)
    : m_input_index(input_index),
      m_body_parameter_index(body_parameter_index) {}

op::util::MultiSubGraphOp::OutputDescription::OutputDescription(uint64_t body_value_index, uint64_t output_index)
    : m_body_value_index(body_value_index),
      m_output_index(output_index) {}

op::util::MultiSubGraphOp::SliceInputDescription::SliceInputDescription(uint64_t input_index,
                                                                        uint64_t body_parameter_index,
                                                                        int64_t start,
                                                                        int64_t stride,
                                                                        int64_t part_size,
                                                                        int64_t end,
                                                                        int64_t axis)
    : InputDescription(input_index, body_parameter_index),
      m_start(start),
      m_stride(stride),
      m_part_size(part_size),
      m_end(end),
      m_axis(axis) {}

std::shared_ptr<op::util::MultiSubGraphOp::InputDescription> op::util::MultiSubGraphOp::SliceInputDescription::copy()
    const {
    return std::make_shared<SliceInputDescription>(m_input_index,
                                                   m_body_parameter_index,
                                                   m_start,
                                                   m_stride,
                                                   m_part_size,
                                                   m_end,
                                                   m_axis);
}

op::util::MultiSubGraphOp::MergedInputDescription::MergedInputDescription(uint64_t input_index,
                                                                          uint64_t body_parameter_index,
                                                                          uint64_t body_value_index)
    : InputDescription(input_index, body_parameter_index),
      m_body_value_index(body_value_index) {}

std::shared_ptr<op::util::MultiSubGraphOp::InputDescription> op::util::MultiSubGraphOp::MergedInputDescription::copy()
    const {
    return std::make_shared<MergedInputDescription>(m_input_index, m_body_parameter_index, m_body_value_index);
}

op::util::MultiSubGraphOp::ConcatOutputDescription::ConcatOutputDescription(uint64_t body_value_index,
                                                                            uint64_t output_index,
                                                                            int64_t start,
                                                                            int64_t stride,
                                                                            int64_t part_size,
                                                                            int64_t end,
                                                                            int64_t axis)
    : OutputDescription(body_value_index, output_index),
      m_start(start),
      m_stride(stride),
      m_part_size(part_size),
      m_end(end),
      m_axis(axis) {}

std::shared_ptr<op::util::MultiSubGraphOp::OutputDescription> op::util::MultiSubGraphOp::ConcatOutputDescription::copy()
    const {
    return std::make_shared<ConcatOutputDescription>(m_body_value_index,
                                                     m_output_index,
                                                     m_start,
                                                     m_stride,
                                                     m_part_size,
                                                     m_end,
                                                     m_axis);
}
op::util::MultiSubGraphOp::InvariantInputDescription::InvariantInputDescription(uint64_t input_index,
                                                                                uint64_t body_parameter_index)
    : InputDescription(input_index, body_parameter_index) {}

std::shared_ptr<op::util::MultiSubGraphOp::InputDescription>
op::util::MultiSubGraphOp::InvariantInputDescription::copy() const {
    return std::make_shared<MultiSubGraphOp::InvariantInputDescription>(m_input_index, m_body_parameter_index);
}

op::util::MultiSubGraphOp::BodyOutputDescription::BodyOutputDescription(uint64_t body_value_index,
                                                                        uint64_t output_index,
                                                                        int64_t iteration)
    : OutputDescription(body_value_index, output_index),
      m_iteration(iteration) {}

std::shared_ptr<op::util::MultiSubGraphOp::OutputDescription> op::util::MultiSubGraphOp::BodyOutputDescription::copy()
    const {
    return std::make_shared<BodyOutputDescription>(m_body_value_index, m_output_index, m_iteration);
}

op::util::MultiSubGraphOp::MultiSubGraphOp(const OutputVector& args) : Op(args) {}

op::util::MultiSubGraphOp::MultiSubGraphOp(size_t number_of_bodies) {
    m_bodies.resize(number_of_bodies);
    m_input_descriptions.resize(number_of_bodies);
    m_output_descriptions.resize(number_of_bodies);
}

op::util::MultiSubGraphOp::MultiSubGraphOp(const OutputVector& args, size_t number_of_bodies) : MultiSubGraphOp(args) {
    m_bodies.resize(number_of_bodies);
    m_input_descriptions.resize(number_of_bodies);
    m_output_descriptions.resize(number_of_bodies);
}

Input<Node> op::util::MultiSubGraphOp::input_for_value(const Output<Node>& value) {
    auto input_index = get_input_size();
    set_argument(input_index, value);
    return Input<Node>(this, input_index);
}

void op::util::MultiSubGraphOp::set_invariant_inputs(const Output<Node>& value,
                                                     const ParameterVector& bodies_parameters) {
    auto input_index = input_for_value(value).get_index();
    for (auto& param : bodies_parameters) {
        for (size_t body_index = 0; body_index < m_bodies.size(); ++body_index) {
            auto param_index = m_bodies[body_index]->get_parameter_index(param);
            if (param_index != -1) {
                m_input_descriptions[body_index].push_back(
                    std::make_shared<MultiSubGraphOp::InvariantInputDescription>(input_index, param_index));
            }
        }
    }
}

Output<Node> op::util::MultiSubGraphOp::set_body_outputs(const ResultVector& bodies_results) {
    auto output_index = get_output_size();
    for (auto& body_result : bodies_results) {
        for (size_t body_index = 0; body_index < m_bodies.size(); body_index++) {
            auto body_result_index = m_bodies[body_index]->get_result_index(body_result);
            if (body_result_index != -1) {
                m_output_descriptions[body_index].push_back(
                    std::make_shared<BodyOutputDescription>(body_result_index, output_index));
            }
        }
    }
    set_output_size(output_index + 1);
    return Output<Node>(shared_from_this(), output_index);
}

namespace ov {
NGRAPH_RTTI_DEFINITION(AttributeAdapter<std::vector<std::shared_ptr<op::util::MultiSubGraphOp::InputDescription>>>,
                       "AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::"
                       "MultiSubGraphOp::InputDescription>>>",
                       0);

NGRAPH_RTTI_DEFINITION(AttributeAdapter<std::vector<std::shared_ptr<op::util::MultiSubGraphOp::OutputDescription>>>,
                       "AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::"
                       "MultiSubGraphOp::OutputDescription>>>",
                       0);
}  // namespace ov
