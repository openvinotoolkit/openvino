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

#include "ngraph/op/region_yolo.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::RegionYolo::type_info;

op::RegionYolo::RegionYolo(const Output<Node>& input,
                           const size_t coords,
                           const size_t classes,
                           const size_t regions,
                           const bool do_softmax,
                           const vector<int64_t>& mask,
                           const int axis,
                           const int end_axis,
                           const vector<float>& anchors)
    : Op({input})
    , m_num_coords(coords)
    , m_num_classes(classes)
    , m_num_regions(regions)
    , m_do_softmax(do_softmax)
    , m_mask(mask)
    , m_anchors(anchors)
    , m_axis(axis)
    , m_end_axis(end_axis)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::RegionYolo::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_RegionYolo_visit_attributes);
    visitor.on_attribute("anchors", m_anchors);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("coords", m_num_coords);
    visitor.on_attribute("classes", m_num_classes);
    visitor.on_attribute("end_axis", m_end_axis);
    visitor.on_attribute("num", m_num_regions);
    visitor.on_attribute("do_softmax", m_do_softmax);
    visitor.on_attribute("mask", m_mask);
    return true;
}

void op::RegionYolo::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_RegionYolo_validate_and_infer_types);
    auto input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_real(),
                          "Type of input is expected to be a floating point type. Got: ",
                          input_et);

    if (get_input_partial_shape(0).is_static())
    {
        Shape input_shape = get_input_partial_shape(0).to_shape();
        Shape output_shape;
        int end_axis = m_end_axis;
        if (m_end_axis < 0)
        {
            m_end_axis += input_shape.size();
        }

        if (m_do_softmax)
        {
            size_t flat_dim = 1;
            for (int64_t i = 0; i < m_axis; i++)
            {
                output_shape.push_back(input_shape[i]);
            }
            for (int64_t i = m_axis; i < end_axis + 1; i++)
            {
                flat_dim *= input_shape[i];
            }
            output_shape.push_back(flat_dim);
            for (size_t i = end_axis + 1; i < input_shape.size(); i++)
            {
                output_shape.push_back(input_shape[i]);
            }
        }
        else
        {
            output_shape = {input_shape[0],
                            (m_num_classes + m_num_coords + 1) * m_mask.size(),
                            input_shape[2],
                            input_shape[3]};
        }
        set_output_type(0, input_et, output_shape);
    }
    else
    {
        set_output_type(0, input_et, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::RegionYolo::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_RegionYolo_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<RegionYolo>(new_args.at(0),
                                   m_num_coords,
                                   m_num_classes,
                                   m_num_regions,
                                   m_do_softmax,
                                   m_mask,
                                   m_axis,
                                   m_end_axis,
                                   m_anchors);
}
