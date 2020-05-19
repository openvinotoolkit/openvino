//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/deformable_convolution.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::DeformableConvolution::type_info;

op::v1::DeformableConvolution::DeformableConvolution(const Output<Node>& arg,
                                                     const Output<Node>& deformable_values,
                                                     const Output<Node>& filters,
                                                     const Strides& strides,
                                                     const CoordinateDiff& pads_begin,
                                                     const CoordinateDiff& pads_end,
                                                     const Strides& dilations,
                                                     const PadType& auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group)
    : Op({arg, deformable_values, filters})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
    , m_group(group)
    , m_deformable_group(deformable_group)
{
    constructor_validate_and_infer_types();
}

bool op::v1::DeformableConvolution::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("group", m_group);
    visitor.on_attribute("deformable_group", m_deformable_group);
    return true;
}

void op::v1::DeformableConvolution::validate_and_infer_types()
{
    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    const PartialShape& deformable_values_shape = get_input_partial_shape(1);
    const PartialShape& filters_shape = get_input_partial_shape(2);

    element::Type data_batch_et = get_input_element_type(0);
    element::Type deformable_values_et = get_input_element_type(1);
    element::Type filters_et = get_input_element_type(2);

    if (m_strides.size() == 0)
    {
        m_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_dilations.size() == 0)
    {
        m_dilations = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_pads_begin.size() == 0)
    {
        m_pads_begin = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_pads_end.size() == 0)
    {
        m_pads_end = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        if (data_batch_shape.is_static() && filters_shape.is_static())
        {
            m_pads_begin.clear();
            m_pads_end.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            infer_auto_padding(data_batch_shape.to_shape(),
                               filter_shape,
                               m_strides,
                               m_dilations,
                               m_auto_pad,
                               m_pads_end,
                               m_pads_begin);
        }
    }

    if (deformable_values_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(
            this,
            deformable_values_shape.rank().get_length() >= 3u,
            "The deformable values tensor rank is expected to be at least 3, got: ",
            deformable_values_shape.rank());
    }

    if (m_group > 1 && data_batch_shape[1].is_static() && filters_shape[0].is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              data_batch_shape[1].get_length() % m_group == 0,
                              "The input data shape must be evenly divisible by the 'group' value "
                              "along the channels axis. Current input shape: ",
                              data_batch_shape,
                              ", 'group' attribute value: ",
                              m_group);

        NODE_VALIDATION_CHECK(
            this,
            filters_shape[0].get_length() % m_group == 0,
            "The weights shape must be evenly divisible by the 'group' value along "
            "the channels axis. Current weights shape: ",
            filters_shape,
            ", 'group' attribute value: ",
            m_group);
    }

    if (m_deformable_group > 1 && deformable_values_shape[1].is_static())
    {
        NODE_VALIDATION_CHECK(
            this,
            deformable_values_shape[1].get_length() % m_deformable_group == 0,
            "The deformable values input must be evenly divisible by the 'deformable group' value "
            "along the channels axis. Current input shape: ",
            deformable_values_shape,
            ", 'deformable group' attribute value: ",
            m_deformable_group);
    }

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_batch_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        filters_et,
        ").");

    const PartialShape result_shape =
        infer_convolution_forward(this,
                                  data_batch_shape,
                                  Strides(m_strides.size(), 1), // dummy data dilations
                                  m_pads_begin,
                                  m_pads_end,
                                  filters_shape,
                                  m_strides,
                                  m_dilations);

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node>
    op::v1::DeformableConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::DeformableConvolution>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  m_strides,
                                                  m_pads_begin,
                                                  m_pads_end,
                                                  m_dilations,
                                                  m_auto_pad,
                                                  m_group,
                                                  m_deformable_group);
}
