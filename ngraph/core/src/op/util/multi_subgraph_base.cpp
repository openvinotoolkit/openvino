// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/multi_subgraph_base.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "ngraph/graph_util.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::MultiSubGraphOp, "MultiSubGraphOp", 0);

constexpr DiscreteTypeInfo op::util::MultiSubGraphOp::InvariantInputDescription::type_info;
constexpr DiscreteTypeInfo op::util::MultiSubGraphOp::BodyOutputDescription::type_info;

op::util::MultiSubGraphOp::InputDescription::InputDescription(uint64_t input_index,
                                                              uint64_t body_parameter_index)
    : m_input_index(input_index)
    , m_body_parameter_index(body_parameter_index)
{
}

op::util::MultiSubGraphOp::OutputDescription::OutputDescription(uint64_t body_value_index,
                                                                uint64_t output_index)
    : m_body_value_index(body_value_index)
    , m_output_index(output_index)
{
}

op::util::MultiSubGraphOp::InvariantInputDescription::InvariantInputDescription(
    uint64_t input_index, uint64_t body_parameter_index)
    : InputDescription(input_index, body_parameter_index)
{
}

std::shared_ptr<op::util::MultiSubGraphOp::InputDescription>
    op::util::MultiSubGraphOp::InvariantInputDescription::copy() const
{
    return std::make_shared<MultiSubGraphOp::InvariantInputDescription>(m_input_index,
                                                                        m_body_parameter_index);
}

op::util::MultiSubGraphOp::BodyOutputDescription::BodyOutputDescription(uint64_t body_value_index,
                                                                        uint64_t output_index)
    : OutputDescription(body_value_index, output_index)
{
}

std::shared_ptr<op::util::MultiSubGraphOp::OutputDescription>
    op::util::MultiSubGraphOp::BodyOutputDescription::copy() const
{
    return std::make_shared<BodyOutputDescription>(m_body_value_index, m_output_index);
}

op::util::MultiSubGraphOp::MultiSubGraphOp(const OutputVector& args)
    : Op(args)
{
}

op::util::MultiSubGraphOp::MultiSubGraphOp(size_t bodies_index)
{
    m_bodies.resize(bodies_index);
    m_input_descriptions.resize(bodies_index);
    m_output_descriptions.resize(bodies_index);
}

op::util::MultiSubGraphOp::MultiSubGraphOp(const OutputVector& args, size_t bodies_index)
    : MultiSubGraphOp(args)
{
    m_bodies.resize(bodies_index);
    m_input_descriptions.resize(bodies_index);
    m_output_descriptions.resize(bodies_index);
}

Input<Node> op::util::MultiSubGraphOp::input_for_value(const Output<Node>& value)
{
    auto input_index = get_input_size();
    set_argument(input_index, value);
    return Input<Node>(this, input_index);
}

void op::util::MultiSubGraphOp::set_invariant_inputs(const Output<Node>& value,
                                                     const ParameterVector bodies_parameters)
{
    auto input_index = input_for_value(value).get_index();
    size_t body_index = 0;
    for (auto& param : bodies_parameters)
    {
        if (param == nullptr)
        {
            body_index++;
            continue;
        }
        m_input_descriptions[body_index].push_back(
            std::make_shared<MultiSubGraphOp::InvariantInputDescription>(
                input_index, m_bodies[body_index]->get_parameter_index(param)));
        body_index++;
    }
}

Output<Node> op::util::MultiSubGraphOp::set_body_outputs(ResultVector bodies_results)
{
    auto output_index = m_output_descriptions[0].size();
    size_t body_index = 0;
    for (auto& body_result : bodies_results)
    {
        m_output_descriptions[body_index].push_back(std::make_shared<BodyOutputDescription>(
            m_bodies[body_index]->get_result_index(body_result), output_index));
        body_index++;
    }
    set_output_size(output_index + 1);
    return Output<Node>(shared_from_this(), output_index);
}

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<
        std::vector<std::shared_ptr<op::util::MultiSubGraphOp::InputDescription>>>::type_info;

    constexpr DiscreteTypeInfo AttributeAdapter<
        std::vector<std::shared_ptr<op::util::MultiSubGraphOp::OutputDescription>>>::type_info;
} // namespace ngraph
