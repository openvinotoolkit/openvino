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
                           const int pooled_h,
                           const int pooled_w,
                           const float spatial_scale,
                           const string& method)
    : Op({input, coords})
    , m_pooled_h(pooled_h)
    , m_pooled_w(pooled_w)
    , m_spatial_scale(spatial_scale)
    , m_method(EnumNames<ROIPooling::ROIPoolingMethod>::as_enum(method))
{
    constructor_validate_and_infer_types();
}

op::ROIPooling::ROIPooling(const Output<Node>& input,
                           const Output<Node>& coords,
                           const int pooled_h,
                           const int pooled_w,
                           const float spatial_scale,
                           ROIPoolingMethod method)
    : Op({input, coords})
    , m_pooled_h(pooled_h)
    , m_pooled_w(pooled_w)
    , m_spatial_scale(spatial_scale)
    , m_method(method)
{
    constructor_validate_and_infer_types();
}

void op::ROIPooling::validate_and_infer_types()
{
    auto input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et.is_real(),
                          "Type of input is expected to be a floating point type. Got: ",
                          input_et);

    if (get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static())
    {
        Shape input_shape = get_input_partial_shape(0).to_shape();
        Shape coords_shape = get_input_partial_shape(1).to_shape();
        NODE_VALIDATION_CHECK(this,
                              input_shape.size() == 4,
                              "ROIPooling expects 4 dimensions for input feature maps. Got ",
                              input_shape.size());
        NODE_VALIDATION_CHECK(this,
                              coords_shape.size() == 2,
                              "ROIPooling expects 2 dimensions for box coordinates. Got ",
                              coords_shape.size());
        Shape output_shape{coords_shape[0],
                           input_shape[1],
                           static_cast<uint64_t>(m_pooled_h),
                           static_cast<uint64_t>(m_pooled_w)};
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
        new_args.at(0), new_args.at(1), m_pooled_h, m_pooled_w, m_spatial_scale, m_method);
}

bool op::ROIPooling::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("pooled_h", m_pooled_h);
    visitor.on_attribute("pooled_w", m_pooled_w);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("method", m_method);
    return true;
}

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<op::v0::ROIPooling::ROIPoolingMethod>::type_info;

    template <>
    EnumNames<op::v0::ROIPooling::ROIPoolingMethod>&
        EnumNames<op::v0::ROIPooling::ROIPoolingMethod>::get()
    {
        static auto enum_names = EnumNames<op::v0::ROIPooling::ROIPoolingMethod>(
            "op::v0::ROIPooling::ROIPoolingMethod",
            {{"bilinear", op::v0::ROIPooling::ROIPoolingMethod::Bilinear},
             {"max", op::v0::ROIPooling::ROIPoolingMethod::Max}});
        return enum_names;
    }

    std::ostream& operator<<(std::ostream& s, const op::v0::ROIPooling::ROIPoolingMethod& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph
