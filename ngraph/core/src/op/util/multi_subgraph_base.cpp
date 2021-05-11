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
    return std::make_shared<InvariantInputDescription>(m_input_index, m_body_parameter_index);
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

void op::util::MultiSubGraphOp::reserve_bodies(int num_bodies)
{
    m_bodies = decltype(m_bodies)(num_bodies);
    m_input_descriptions = decltype(m_input_descriptions)(num_bodies);
    m_output_descriptions = decltype(m_output_descriptions)(num_bodies);
}
namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<
        std::vector<std::shared_ptr<op::util::MultiSubGraphOp::InputDescription>>>::type_info;

    constexpr DiscreteTypeInfo AttributeAdapter<
        std::vector<std::shared_ptr<op::util::MultiSubGraphOp::OutputDescription>>>::type_info;
} // namespace ngraph
