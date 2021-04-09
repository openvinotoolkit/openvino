// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/sub_graph_utils.h"

using namespace ngraph;

constexpr DiscreteTypeInfo op::util::sub_graph_utils::SliceInputDescription::type_info;
constexpr DiscreteTypeInfo op::util::sub_graph_utils::MergedInputDescription::type_info;
constexpr DiscreteTypeInfo op::util::sub_graph_utils::InvariantInputDescription::type_info;

constexpr DiscreteTypeInfo op::util::sub_graph_utils::BodyOutputDescription::type_info;
constexpr DiscreteTypeInfo op::util::sub_graph_utils::ConcatOutputDescription::type_info;

op::util::sub_graph_utils::InputDescription::InputDescription(uint64_t input_index,
                                                         uint64_t body_parameter_index)
    : m_input_index(input_index)
    , m_body_parameter_index(body_parameter_index)
{
}

op::util::sub_graph_utils::SliceInputDescription::SliceInputDescription(uint64_t input_index,
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

std::shared_ptr<op::util::sub_graph_utils::InputDescription>
    op::util::sub_graph_utils::SliceInputDescription::copy() const
{
    return std::make_shared<SliceInputDescription>(
        m_input_index, m_body_parameter_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::util::sub_graph_utils::MergedInputDescription::MergedInputDescription(
    uint64_t input_index,
                                                                     uint64_t body_parameter_index,
                                                                     uint64_t body_value_index)
    : InputDescription(input_index, body_parameter_index)
    , m_body_value_index(body_value_index)
{
}

std::shared_ptr<op::util::sub_graph_utils::InputDescription>
    op::util::sub_graph_utils::MergedInputDescription::copy() const
{
    return std::make_shared<MergedInputDescription>(
        m_input_index, m_body_parameter_index, m_body_value_index);
}

op::util::sub_graph_utils::InvariantInputDescription::InvariantInputDescription(
    uint64_t input_index, uint64_t body_parameter_index)
    : InputDescription(input_index, body_parameter_index)
{
}

std::shared_ptr<op::util::sub_graph_utils::InputDescription>
    op::util::sub_graph_utils::InvariantInputDescription::copy() const
{
    return std::make_shared<InvariantInputDescription>(m_input_index, m_body_parameter_index);
}

op::util::sub_graph_utils::OutputDescription::OutputDescription(uint64_t body_value_index,
                                                           uint64_t output_index)
    : m_body_value_index(body_value_index)
    , m_output_index(output_index)
{
}

op::util::sub_graph_utils::ConcatOutputDescription::ConcatOutputDescription(uint64_t body_value_index,
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

std::shared_ptr<op::util::sub_graph_utils::OutputDescription>
    op::util::sub_graph_utils::ConcatOutputDescription::copy() const
{
    return std::make_shared<ConcatOutputDescription>(
        m_body_value_index, m_output_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::util::sub_graph_utils::BodyOutputDescription::BodyOutputDescription(uint64_t body_value_index,
                                                                   uint64_t output_index,
                                                                   int64_t iteration)
    : OutputDescription(body_value_index, output_index)
    , m_iteration(iteration)
{
}

std::shared_ptr<op::util::sub_graph_utils::OutputDescription>
    op::util::sub_graph_utils::BodyOutputDescription::copy() const
{
    return std::make_shared<BodyOutputDescription>(m_body_value_index, m_output_index, m_iteration);
}