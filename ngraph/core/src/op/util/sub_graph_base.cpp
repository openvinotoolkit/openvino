// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/sub_graph_base.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "ngraph/graph_util.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::SubGraphOp, "SubGraphOp", 0);

constexpr DiscreteTypeInfo op::util::SubGraphOp::SliceInputDescription::type_info;
constexpr DiscreteTypeInfo op::util::SubGraphOp::MergedInputDescription::type_info;
constexpr DiscreteTypeInfo op::util::SubGraphOp::InvariantInputDescription::type_info;

constexpr DiscreteTypeInfo op::util::SubGraphOp::BodyOutputDescription::type_info;
constexpr DiscreteTypeInfo op::util::SubGraphOp::ConcatOutputDescription::type_info;

op::util::SubGraphOp::InputDescription::InputDescription(uint64_t input_index,
                                                         uint64_t body_parameter_index)
    : m_input_index(input_index)
    , m_body_parameter_index(body_parameter_index)
{
}

op::util::SubGraphOp::SliceInputDescription::SliceInputDescription(uint64_t input_index,
                                                                   uint64_t body_parameter_index,
                                                                   int64_t start,
                                                                   int64_t stride,
                                                                   int64_t part_size,
                                                                   int64_t end,
                                                                   int64_t axis)
    : InputDescription(input_index, body_parameter_index)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
    , m_axis(axis)
{
}

std::shared_ptr<op::util::SubGraphOp::InputDescription>
    op::util::SubGraphOp::SliceInputDescription::copy() const
{
    return std::make_shared<SliceInputDescription>(
        m_input_index, m_body_parameter_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::util::SubGraphOp::MergedInputDescription::MergedInputDescription(uint64_t input_index,
                                                                     uint64_t body_parameter_index,
                                                                     uint64_t body_value_index)
    : InputDescription(input_index, body_parameter_index)
    , m_body_value_index(body_value_index)
{
}

std::shared_ptr<op::util::SubGraphOp::InputDescription>
    op::util::SubGraphOp::MergedInputDescription::copy() const
{
    return std::make_shared<MergedInputDescription>(
        m_input_index, m_body_parameter_index, m_body_value_index);
}

op::util::SubGraphOp::InvariantInputDescription::InvariantInputDescription(
    uint64_t input_index, uint64_t body_parameter_index)
    : InputDescription(input_index, body_parameter_index)
{
}

std::shared_ptr<op::util::SubGraphOp::InputDescription>
    op::util::SubGraphOp::InvariantInputDescription::copy() const
{
    return std::make_shared<InvariantInputDescription>(m_input_index, m_body_parameter_index);
}

op::util::SubGraphOp::OutputDescription::OutputDescription(uint64_t body_value_index,
                                                           uint64_t output_index)
    : m_body_value_index(body_value_index)
    , m_output_index(output_index)
{
}

op::util::SubGraphOp::ConcatOutputDescription::ConcatOutputDescription(uint64_t body_value_index,
                                                                       uint64_t output_index,
                                                                       int64_t start,
                                                                       int64_t stride,
                                                                       int64_t part_size,
                                                                       int64_t end,
                                                                       int64_t axis)
    : OutputDescription(body_value_index, output_index)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
    , m_axis(axis)
{
}

std::shared_ptr<op::util::SubGraphOp::OutputDescription>
    op::util::SubGraphOp::ConcatOutputDescription::copy() const
{
    return std::make_shared<ConcatOutputDescription>(
        m_body_value_index, m_output_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::util::SubGraphOp::BodyOutputDescription::BodyOutputDescription(uint64_t body_value_index,
                                                                   uint64_t output_index,
                                                                   int64_t iteration)
    : OutputDescription(body_value_index, output_index)
    , m_iteration(iteration)
{
}

std::shared_ptr<op::util::SubGraphOp::OutputDescription>
    op::util::SubGraphOp::BodyOutputDescription::copy() const
{
    return std::make_shared<BodyOutputDescription>(m_body_value_index, m_output_index, m_iteration);
}

op::util::SubGraphOp::SubGraphOp(const OutputVector& args)
    : Op(args)
{
}

void op::util::SubGraphOp::set_merged_input(const std::shared_ptr<Parameter>& body_parameter,
                                            const Output<Node>& initial_value,
                                            const Output<Node>& successive_value)
{
    m_input_descriptions.push_back(std::make_shared<TensorIterator::MergedInputDescription>(
        input_for_value(initial_value).get_index(),
        m_body->get_parameter_index(body_parameter),
        m_body->get_result_index(successive_value)));
    validate_and_infer_types();
}

void op::util::SubGraphOp::set_invariant_input(const std::shared_ptr<Parameter>& body_parameter,
                                               const Output<Node>& value)
{
    m_input_descriptions.push_back(std::make_shared<TensorIterator::InvariantInputDescription>(
        input_for_value(value).get_index(), m_body->get_parameter_index(body_parameter)));
    validate_and_infer_types();
}

Output<Node> op::util::SubGraphOp::get_iter_value(const Output<Node>& body_value, int64_t iteration)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(std::make_shared<BodyOutputDescription>(
        m_body->get_result_index(body_value), output_index, iteration));
    set_output_size(output_index + 1);
    validate_and_infer_types();
    return Output<Node>(shared_from_this(), output_index);
}

Output<Node> op::util::SubGraphOp::get_concatenated_slices(const Output<Node>& body_value,
                                                           int64_t start,
                                                           int64_t stride,
                                                           int64_t part_size,
                                                           int64_t end,
                                                           int64_t axis)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(std::make_shared<ConcatOutputDescription>(
        m_body->get_result_index(body_value), output_index, start, stride, part_size, end, axis));
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
                                            int64_t axis)
{
    m_input_descriptions.push_back(
        std::make_shared<SliceInputDescription>(input_for_value(value).get_index(),
                                                m_body->get_parameter_index(parameter),
                                                start,
                                                stride,
                                                part_size,
                                                end,
                                                axis));
    validate_and_infer_types();
}

Input<Node> op::util::SubGraphOp::input_for_value(const Output<Node>& value)
{
    auto input_index = get_input_size();
    set_argument(input_index, value);
    return Input<Node>(this, input_index);
}

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<
        std::vector<std::shared_ptr<op::util::SubGraphOp::InputDescription>>>::type_info;

    constexpr DiscreteTypeInfo AttributeAdapter<
        std::vector<std::shared_ptr<op::util::SubGraphOp::OutputDescription>>>::type_info;
} // namespace ngraph
