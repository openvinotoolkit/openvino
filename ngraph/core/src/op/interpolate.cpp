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

#include "ngraph/op/interpolate.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Interpolate::type_info;

op::v0::Interpolate::Interpolate(const Output<Node>& image,
                                 const Output<Node>& output_shape,
                                 const op::v0::InterpolateAttrs& attrs)
    : Op({image, output_shape})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

bool op::v0::Interpolate::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("attrs", m_attrs);
    return true;
}

void op::v0::Interpolate::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "output shape must be an integral number.");
    set_input_is_relevant_to_shape(1);

    PartialShape output_shape = PartialShape(get_input_partial_shape(0));
    if (output_shape.rank().is_static())
    {
        for (auto axis : m_attrs.axes)
        {
            NGRAPH_CHECK(axis < output_shape.rank().get_length());
            output_shape[axis] = Dimension::dynamic();
        }
    }

    if (auto const_shape = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        auto out_shape = const_shape->cast_vector<int64_t>();
        size_t i = 0;
        for (auto axis : m_attrs.axes)
        {
            output_shape[axis] = Dimension(out_shape[i++]);
        }
    }
    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::v0::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Interpolate>(new_args.at(0), new_args.at(1), m_attrs);
}

namespace ngraph
{
    template <>
    EnumNames<op::v0::Interpolate::InterpolateMode>&
        EnumNames<op::v0::Interpolate::InterpolateMode>::get()
    {
        static auto enum_names = EnumNames<op::v0::Interpolate::InterpolateMode>(
            "op::v0::Interpolate::InterpolateMode",
            {{"nearest", op::v0::Interpolate::InterpolateMode::nearest},
             {"linear", op::v0::Interpolate::InterpolateMode::linear},
             {"cubic", op::v0::Interpolate::InterpolateMode::cubic},
             {"area", op::v0::Interpolate::InterpolateMode::area}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v0::Interpolate::InterpolateMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v0::Interpolate::InterpolateMode& type)
    {
        return s << as_string(type);
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v0::InterpolateAttrs>::type_info;

    AttributeAdapter<op::v0::InterpolateAttrs>::AttributeAdapter(op::v0::InterpolateAttrs& ref)
        : m_ref(ref)
    {
    }

    bool AttributeAdapter<op::v0::InterpolateAttrs>::visit_attributes(AttributeVisitor& visitor)
    {
        visitor.on_attribute("axes", m_ref.axes);
        visitor.on_attribute("mode", m_ref.mode);
        visitor.on_attribute("align_corners", m_ref.align_corners);
        visitor.on_attribute("antialias", m_ref.antialias);
        visitor.on_attribute("pads_begin", m_ref.pads_begin);
        visitor.on_attribute("pads_end", m_ref.pads_end);
        return true;
    }
}

// Interpolate v3

constexpr NodeTypeInfo op::v3::Interpolate::type_info;

op::v3::Interpolate::Interpolate(const Output<Node>& image,
                                 const Output<Node>& output_shape,
                                 const op::v3::Interpolate::InterpolateAttrs& attrs)
    : Op({image, output_shape})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

bool op::v3::Interpolate::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("attrs", m_attrs);
    return true;
}

void op::v3::Interpolate::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "output shape must be an integral number.");
    set_input_is_relevant_to_shape(1);

    PartialShape output_shape = PartialShape(get_input_partial_shape(0));
    if (output_shape.rank().is_static())
    {
        for (auto axis : m_attrs.axes)
        {
            NGRAPH_CHECK(axis < output_shape.rank().get_length());
            output_shape[axis] = Dimension::dynamic();
        }
    }

    if (auto const_shape = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        auto out_shape = const_shape->cast_vector<int64_t>();
        size_t i = 0;
        for (auto axis : m_attrs.axes)
        {
            output_shape[axis] = Dimension(out_shape[i++]);
        }
    }
    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::v3::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v3::Interpolate>(new_args.at(0), new_args.at(1), m_attrs);
}

namespace ngraph
{
    template <>
    NGRAPH_API EnumNames<op::v3::Interpolate::InterpolateMode>&
        EnumNames<op::v3::Interpolate::InterpolateMode>::get()
    {
        static auto enum_names = EnumNames<op::v3::Interpolate::InterpolateMode>(
            "op::v3::Interpolate::InterpolateMode",
            {{"nearest", op::v3::Interpolate::InterpolateMode::nearest},
             {"linear", op::v3::Interpolate::InterpolateMode::linear},
             {"linear_onnx", op::v3::Interpolate::InterpolateMode::linear_onnx},
             {"cubic", op::v3::Interpolate::InterpolateMode::cubic},
             {"area", op::v3::Interpolate::InterpolateMode::area}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v3::Interpolate::InterpolateMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v3::Interpolate::InterpolateMode& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::v3::Interpolate::CoordinateTransformMode>&
        EnumNames<op::v3::Interpolate::CoordinateTransformMode>::get()
    {
        static auto enum_names = EnumNames<op::v3::Interpolate::CoordinateTransformMode>(
            "op::v3::Interpolate::CoordinateTransformMode",
            {{"half_pixel", op::v3::Interpolate::CoordinateTransformMode::half_pixel},
             {"pytorch_half_pixel",
              op::v3::Interpolate::CoordinateTransformMode::pytorch_half_pixel},
             {"asymmetric", op::v3::Interpolate::CoordinateTransformMode::asymmetric},
             {"tf_half_pixel_for_nn",
              op::v3::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn},
             {"align_corners", op::v3::Interpolate::CoordinateTransformMode::align_corners}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<op::v3::Interpolate::CoordinateTransformMode>::type_info;

    std::ostream& operator<<(std::ostream& s,
                             const op::v3::Interpolate::CoordinateTransformMode& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::v3::Interpolate::NearestMode>& EnumNames<op::v3::Interpolate::NearestMode>::get()
    {
        static auto enum_names = EnumNames<op::v3::Interpolate::NearestMode>(
            "op::v3::Interpolate::NearestMode",
            {{"round_prefer_floor", op::v3::Interpolate::NearestMode::round_prefer_floor},
             {"round_prefer_ceil", op::v3::Interpolate::NearestMode::round_prefer_ceil},
             {"floor", op::v3::Interpolate::NearestMode::floor},
             {"ceil", op::v3::Interpolate::NearestMode::ceil},
             {"simple", op::v3::Interpolate::NearestMode::simple}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v3::Interpolate::NearestMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v3::Interpolate::NearestMode& type)
    {
        return s << as_string(type);
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v3::Interpolate::InterpolateAttrs>::type_info;

    AttributeAdapter<op::v3::Interpolate::InterpolateAttrs>::AttributeAdapter(
        op::v3::Interpolate::InterpolateAttrs& ref)
        : m_ref(ref)
    {
    }

    bool AttributeAdapter<op::v3::Interpolate::InterpolateAttrs>::visit_attributes(
        AttributeVisitor& visitor)
    {
        visitor.on_attribute("axes", m_ref.axes);
        visitor.on_attribute("mode", m_ref.mode);
        visitor.on_attribute("coordinate_transformation_mode",
                             m_ref.coordinate_transformation_mode);
        visitor.on_attribute("nearest_mode", m_ref.nearest_mode);
        visitor.on_attribute("antialias", m_ref.antialias);
        visitor.on_attribute("pads_begin", m_ref.pads_begin);
        visitor.on_attribute("pads_end", m_ref.pads_end);
        visitor.on_attribute("cube_coeff", m_ref.cube_coeff);
        return true;
    }
}
