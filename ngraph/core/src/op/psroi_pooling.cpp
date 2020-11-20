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

#include "ngraph/op/psroi_pooling.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PSROIPooling::type_info;

op::PSROIPooling::PSROIPooling(const Output<Node>& input,
                               const Output<Node>& coords,
                               const size_t output_dim,
                               const size_t group_size,
                               const float spatial_scale,
                               int spatial_bins_x,
                               int spatial_bins_y,
                               const string& mode)
    : Op({input, coords})
    , m_output_dim(output_dim)
    , m_group_size(group_size)
    , m_spatial_scale(spatial_scale)
    , m_spatial_bins_x(spatial_bins_x)
    , m_spatial_bins_y(spatial_bins_y)
    , m_mode(mode)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::PSROIPooling::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("output_dim", m_output_dim);
    visitor.on_attribute("group_size", m_group_size);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("spatial_bins_x", m_spatial_bins_x);
    visitor.on_attribute("spatial_bins_y", m_spatial_bins_y);
    return true;
}

void op::PSROIPooling::validate_and_infer_types()
{
    auto feat_maps_et = get_input_element_type(0);
    auto coords_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          feat_maps_et.is_real(),
                          "Feature maps' data type must be floating point. Got " +
                              feat_maps_et.get_type_name());
    NODE_VALIDATION_CHECK(this,
                          coords_et.is_real(),
                          "Coords' data type must be floating point. Got " +
                              coords_et.get_type_name());
    NODE_VALIDATION_CHECK(this,
                          m_mode == "average" || m_mode == "bilinear",
                          "Expected 'average' or 'bilinear' mode. Got " + m_mode);
    NODE_VALIDATION_CHECK(this, m_group_size > 0, "group_size has to be greater than 0");
    if (get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static())
    {
        Shape input_shape = get_input_partial_shape(0).to_shape();
        Shape coords_shape = get_input_partial_shape(1).to_shape();
        NODE_VALIDATION_CHECK(this,
                              input_shape.size() == 4,
                              "PSROIPooling expects 4 dimensions for input. Got ",
                              input_shape.size());
        NODE_VALIDATION_CHECK(this,
                              coords_shape.size() == 2,
                              "PSROIPooling expects 2 dimensions for box coordinates. Got ",
                              coords_shape.size());
        if (m_mode == "average")
        {
            NODE_VALIDATION_CHECK(
                this,
                input_shape[1] % (m_group_size * m_group_size) == 0,
                "Number of input's channels must be a multiply of group_size * group_size");
            NODE_VALIDATION_CHECK(
                this,
                m_output_dim == input_shape[1] / (m_group_size * m_group_size),
                "output_dim must be equal to input channels divided by group_size * group_size");
        }
        else if (m_mode == "bilinear")
        {
            NODE_VALIDATION_CHECK(
                this, m_spatial_bins_x > 0, "spatial_bins_x has to be greater than 0");
            NODE_VALIDATION_CHECK(
                this, m_spatial_bins_y > 0, "spatial_bins_y has to be greater than 0");
            NODE_VALIDATION_CHECK(
                this,
                input_shape[1] % (m_spatial_bins_x * m_spatial_bins_y) == 0,
                "Number of input's channels must be a multiply of spatial_bins_x * spatial_bins_y");
            NODE_VALIDATION_CHECK(this,
                                  m_output_dim ==
                                      input_shape[1] / (m_spatial_bins_x * m_spatial_bins_y),
                                  "output_dim must be equal to input channels divided by "
                                  "spatial_bins_x * spatial_bins_y");
        }
        Shape output_shape{coords_shape[0], m_output_dim};
        for (size_t i = 2; i < input_shape.size(); i++)
        {
            output_shape.push_back(m_group_size);
        }
        set_output_type(0, feat_maps_et, output_shape);
    }
    else
    {
        set_output_type(0, feat_maps_et, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::PSROIPooling::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<PSROIPooling>(new_args.at(0),
                                     new_args.at(1),
                                     m_output_dim,
                                     m_group_size,
                                     m_spatial_scale,
                                     m_spatial_bins_x,
                                     m_spatial_bins_y,
                                     m_mode);
}
