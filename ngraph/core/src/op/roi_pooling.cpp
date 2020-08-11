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

#include "ngraph/op/roi_pooling.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ROIPooling::type_info;

op::ROIPooling::ROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const Shape& output_size,
                           const float spatial_scale,
                           const string& method)
    : Op({input, coords})
    , m_output_size(output_size)
    , m_spatial_scale(spatial_scale)
    , m_method(method)
{
    constructor_validate_and_infer_types();
}

void op::ROIPooling::validate_and_infer_types()
{
    auto input_et = get_input_element_type(0);
    if (get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static())
    {
        Shape input_shape = get_input_partial_shape(0).to_shape();
        Shape coords_shape = get_input_partial_shape(1).to_shape();
        NODE_VALIDATION_CHECK(this,
                              input_shape.size() >= 3,
                              "ROIPooling expects 3 or higher dimensions for input. Got ",
                              input_shape.size());
        NODE_VALIDATION_CHECK(this,
                              coords_shape.size() == 2,
                              "ROIPooling expects 2 dimensions for box coordinates. Got ",
                              coords_shape.size());
        NODE_VALIDATION_CHECK(this,
                              input_shape.size() - 2 == m_output_size.size(),
                              "Spatial dimensions on input: ",
                              input_shape.size() - 2,
                              " doesn't match dimensions on requested output_size: ",
                              m_output_size.size());
        Shape output_shape{coords_shape[0], input_shape[1]};
        output_shape.insert(output_shape.end(), m_output_size.begin(), m_output_size.end());
        set_output_type(0, input_et, output_shape);
    }
    else
    {
        set_output_type(0, input_et, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::ROIPooling::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ROIPooling>(
        new_args.at(0), new_args.at(1), m_output_size, m_spatial_scale, m_method);
}

bool op::ROIPooling::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("output_size", m_output_size);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("method", m_method);
    return true;
}
