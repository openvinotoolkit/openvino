//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "itt.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
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
    NGRAPH_OP_SCOPE(v1_DeformableConvolution_visit_attributes);
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
    NGRAPH_OP_SCOPE(v1_DeformableConvolution_validate_and_infer_types);
    const PartialShape& data_batch_pshape = get_input_partial_shape(0);
    const PartialShape& deformable_values_pshape = get_input_partial_shape(1);
    const PartialShape& filters_pshape = get_input_partial_shape(2);

    element::Type data_batch_et = get_input_element_type(0);
    element::Type deformable_values_et = get_input_element_type(1);
    element::Type filters_et = get_input_element_type(2);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, data_batch_et, deformable_values_et),
                          "Element types for data batch and deformable values do not match (data "
                          "batch element type: ",
                          data_batch_et,
                          ", deformable offsets element type: ",
                          deformable_values_et,
                          ").");

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_batch_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        filters_et,
        ").");

    NODE_VALIDATION_CHECK(
        this, result_et.is_real(), "Element types must be float point. Got: ", result_et);

    NODE_VALIDATION_CHECK(this,
                          data_batch_pshape.rank().compatible(4),
                          "Data batch input must be of rank 4. Got: ",
                          data_batch_pshape);

    NODE_VALIDATION_CHECK(this,
                          filters_pshape.rank().compatible(4),
                          "Filters input must be of rank 4. Got: ",
                          filters_pshape);

    NODE_VALIDATION_CHECK(this,
                          deformable_values_pshape.rank().compatible(4),
                          "Deformable values input must be of rank 4. Got: ",
                          filters_pshape);

    NODE_VALIDATION_CHECK(
        this, m_group > 0, "Attribute 'group' must be any value starting from 1. Got: ", m_group);

    NODE_VALIDATION_CHECK(this,
                          m_deformable_group > 0,
                          "Attribute 'deformable group' must be any value starting from 1. Got: ",
                          m_deformable_group);

    if (deformable_values_pshape.rank().is_static())
    {
        if (deformable_values_pshape[1].is_static())
        {
            if (filters_pshape.rank().is_static() && filters_pshape[2].is_static() &&
                filters_pshape[3].is_static())
            {
                auto deformable_channels = m_deformable_group * filters_pshape[2].get_length() *
                                           filters_pshape[3].get_length() * 2;
                NODE_VALIDATION_CHECK(this,
                                      deformable_values_pshape[1].get_length() ==
                                          deformable_channels,
                                      "The channels dimension of deformable values input is not "
                                      "compatible with filters and 'deformable group' attribute. "
                                      "Deformable values input shape: ",
                                      deformable_values_pshape,
                                      ", deformable 'group' attribute value: ",
                                      m_deformable_group,
                                      ", filters shape: ",
                                      filters_pshape);
            }
            else
            {
                // At least we can check that deformable channels is evenly divisible by deformable
                // group attribute
                NODE_VALIDATION_CHECK(
                    this,
                    deformable_values_pshape[1].get_length() % m_deformable_group == 0,
                    "The channels dimension of deformable values input must be "
                    "evenly divisible by the 'deformable group' value along the "
                    "channels axis. Deformable values input shape: ",
                    deformable_values_pshape,
                    ", 'deformable group' attribute value: ",
                    m_deformable_group);
            }
        }

        if (data_batch_pshape.rank().is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                deformable_values_pshape[0].compatible(data_batch_pshape[0]),
                "Data batch and deformable values batch dimension must be same value. Got: ",
                deformable_values_pshape[0],
                " and ",
                data_batch_pshape[0]);
        }
    }

    if (data_batch_pshape.rank().is_static() && data_batch_pshape[1].is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              data_batch_pshape[1].get_length() % m_group == 0,
                              "The input data shape must be evenly divisible by the 'group' value "
                              "along the channels axis. Current input shape: ",
                              data_batch_pshape,
                              ", 'group' attribute value: ",
                              m_group);
    }

    if (filters_pshape.rank().is_static() && filters_pshape[0].is_static())
    {
        NODE_VALIDATION_CHECK(
            this,
            filters_pshape[0].get_length() % m_group == 0,
            "The filters shape must be evenly divisible by the 'group' value along "
            "the channels axis. Current weights shape: ",
            filters_pshape,
            ", 'group' attribute value: ",
            m_group);
    }

    PartialShape result_shape = PartialShape::dynamic();
    Rank output_ps_rank{};
    Rank::merge(output_ps_rank, data_batch_pshape.rank(), filters_pshape.rank());
    Rank::merge(output_ps_rank, output_ps_rank, deformable_values_pshape.rank());
    if (output_ps_rank.is_static())
    {
        const auto num_spatial_dims = output_ps_rank.get_length() - 2;
        if (m_strides.size() == 0)
        {
            m_strides = Strides(num_spatial_dims, 1);
        }

        if (m_dilations.size() == 0)
        {
            m_dilations = Strides(num_spatial_dims, 1);
        }

        if (m_pads_begin.size() == 0 || m_auto_pad == PadType::VALID)
        {
            m_pads_begin = CoordinateDiff(num_spatial_dims, 0);
        }

        if (m_pads_end.size() == 0 || m_auto_pad == PadType::VALID)
        {
            m_pads_end = CoordinateDiff(num_spatial_dims, 0);
        }

        NODE_VALIDATION_CHECK(this,
                              m_strides.size() == num_spatial_dims,
                              "Strides should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_dilations.size() == num_spatial_dims,
                              "Dilations should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              m_pads_begin.size() == num_spatial_dims &&
                                  m_pads_end.size() == num_spatial_dims,
                              "Pads should be defined for all and only spatial features.");

        result_shape = PartialShape::dynamic(output_ps_rank);
        if (data_batch_pshape.rank().is_static() && data_batch_pshape[0].is_static())
        {
            result_shape[0] = data_batch_pshape[0]; // batch size
        }
        if (filters_pshape.rank().is_static())
        {
            result_shape[1] = filters_pshape[0]; // filter channel size
        }

        if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
        {
            bool auto_padding_applied = false;
            if (filters_pshape.rank().is_static() && filters_pshape.rank().get_length() > 2)
            {
                m_pads_begin.clear();
                m_pads_end.clear();
                const PartialShape filter_spatial_shape = [filters_pshape]() {
                    vector<Dimension> filter_dims{filters_pshape};
                    filter_dims.erase(filter_dims.begin(),
                                      filter_dims.begin() + 2); // Remove {C_OUT, C_IN}
                    return PartialShape{filter_dims};
                }();
                if (filter_spatial_shape.is_static())
                {
                    auto_padding_applied = try_apply_auto_padding(data_batch_pshape,
                                                                  filter_spatial_shape.to_shape(),
                                                                  m_strides,
                                                                  m_dilations,
                                                                  m_auto_pad,
                                                                  m_pads_end,
                                                                  m_pads_begin);
                }
            }
            if (!auto_padding_applied)
            {
                set_output_type(0, result_et, result_shape);
                return;
            }
        }
        result_shape =
            infer_convolution_forward(this,
                                      data_batch_pshape,
                                      Strides(m_strides.size(), 1), // dummy data dilations
                                      m_pads_begin,
                                      m_pads_end,
                                      filters_pshape,
                                      m_strides,
                                      m_dilations);

        if (result_shape[0].is_dynamic() && deformable_values_pshape.rank().is_static())
        {
            result_shape[0] = deformable_values_pshape[0]; // batch size
        }
    }
    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node>
    op::v1::DeformableConvolution::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_DeformableConvolution_clone_with_new_inputs);
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
