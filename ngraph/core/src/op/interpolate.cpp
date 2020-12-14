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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/reference/interpolate.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v0::Interpolate, "Interpolate", 0);

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
    visitor.on_attribute("align_corners", m_attrs.align_corners);
    visitor.on_attribute("antialias", m_attrs.antialias);
    visitor.on_attribute("axes", m_attrs.axes);
    visitor.on_attribute("mode", m_attrs.mode);
    visitor.on_attribute("pads_begin", m_attrs.pads_begin);
    visitor.on_attribute("pads_end", m_attrs.pads_end);
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

    if (auto const_shape = as_type_ptr<op::v0::Constant>(input_value(1).get_node_shared_ptr()))
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
} // namespace ngraph

// Interpolate v4

NGRAPH_RTTI_DEFINITION(op::v4::Interpolate, "Interpolate", 4);

op::v4::Interpolate::Interpolate(const Output<Node>& image,
                                 const Output<Node>& output_shape,
                                 const Output<Node>& scales,
                                 const Output<Node>& axes,
                                 const op::v4::Interpolate::InterpolateAttrs& attrs)
    : Op({image, output_shape, scales, axes})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

op::v4::Interpolate::Interpolate(const Output<Node>& image,
                                 const Output<Node>& output_shape,
                                 const Output<Node>& scales,
                                 const op::v4::Interpolate::InterpolateAttrs& attrs)
    : Op({image, output_shape, scales})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

bool op::v4::Interpolate::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("mode", m_attrs.mode);
    visitor.on_attribute("shape_calculation_mode", m_attrs.shape_calculation_mode);
    visitor.on_attribute("coordinate_transformation_mode", m_attrs.coordinate_transformation_mode);
    visitor.on_attribute("nearest_mode", m_attrs.nearest_mode);
    visitor.on_attribute("antialias", m_attrs.antialias);
    visitor.on_attribute("pads_begin", m_attrs.pads_begin);
    visitor.on_attribute("pads_end", m_attrs.pads_end);
    visitor.on_attribute("cube_coeff", m_attrs.cube_coeff);
    return true;
}

std::vector<int64_t> op::v4::Interpolate::get_axes() const
{
    auto inputs = input_values();
    if (inputs.size() <= 3)
    {
        PartialShape input_shape = PartialShape(get_input_partial_shape(0));
        NODE_VALIDATION_CHECK(this,
                              input_shape.rank().is_static(),
                              "Could not define axes of interpolation because there are "
                              "only three inputs and input data has a dynamic rank.");

        const auto input_rank = input_shape.rank().get_length();
        std::vector<int64_t> default_value(input_rank);
        std::iota(default_value.begin(), default_value.end(), 0);

        return default_value;
    }

    auto axes_node = as_type_ptr<op::v0::Constant>(input_value(3).get_node_shared_ptr());
    NODE_VALIDATION_CHECK(this, axes_node, "Input 'axes' should be Constant.");

    return axes_node->cast_vector<int64_t>();
}

static constexpr float epsilon = 1.0e-6f;

void op::v4::Interpolate::infer_using_scales(PartialShape& output_shape,
                                             const std::vector<int64_t>& axes,
                                             const std::vector<float>& scales,
                                             const PartialShape& padded_input_shape) const
{
    size_t i = 0;
    for (auto axis : axes)
    {
        if (padded_input_shape[axis].is_static())
        {
            float padded_len = static_cast<float>(padded_input_shape[axis].get_length());
            int64_t new_dim = static_cast<int64_t>(padded_len * scales[i] + epsilon);
            output_shape[axis] = Dimension(new_dim);
        }
        ++i;
    }
}

void op::v4::Interpolate::infer_using_shapes(PartialShape& output_shape,
                                             const std::vector<int64_t>& axes,
                                             const std::vector<int64_t>& sizes) const
{
    size_t i = 0;
    for (auto axis : axes)
    {
        output_shape[axis] = Dimension(sizes[i++]);
    }
}

PartialShape op::v4::Interpolate::get_padded_input_shape(const PartialShape& input_shape) const
{
    const auto input_rank = input_shape.rank().get_length();

    PartialShape padded_input_shape = input_shape;

    for (size_t i = 0; i < input_rank; ++i)
    {
        if (input_shape[i].is_static())
        {
            auto new_length =
                m_attrs.pads_begin[i] + m_attrs.pads_end[i] + input_shape[i].get_length();
            padded_input_shape[i] = Dimension(new_length);
        }
    }

    return padded_input_shape;
}

void op::v4::Interpolate::validate_and_infer_types()
{
    element::Type input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et == element::f32 || input_et == element::f16 ||
                              input_et == element::i8 || input_et == element::bf16,
                          "Input element type must be f32, f16, bf16 or i8");

    PartialShape input_shape = PartialShape(get_input_partial_shape(0));

    if (!input_shape.rank().is_static())
    {
        set_output_type(0, get_input_element_type(0), input_shape);
        return;
    }

    auto axes = get_axes();
    correct_pads();

    const auto input_rank = input_shape.rank().get_length();

    PartialShape padded_input_shape = get_padded_input_shape(input_shape);
    PartialShape output_shape = padded_input_shape;

    if (output_shape.rank().is_static())
    {
        for (auto axis : axes)
        {
            NGRAPH_CHECK(axis < input_rank);
            output_shape[axis] = Dimension::dynamic();
        }
    }

    set_output_type(0, get_input_element_type(0), output_shape);
    if (m_attrs.shape_calculation_mode == ShapeCalcMode::scales)
    {
        if (auto const_scales = as_type_ptr<op::v0::Constant>(input_value(2).get_node_shared_ptr()))
        {
            auto scales = const_scales->cast_vector<float>();
            infer_using_scales(output_shape, axes, scales, padded_input_shape);
        }
    }
    else
    {
        if (auto const_shape = as_type_ptr<op::v0::Constant>(input_value(1).get_node_shared_ptr()))
        {
            auto sizes = const_shape->cast_vector<int64_t>();
            infer_using_shapes(output_shape, axes, sizes);
        }
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::v4::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() <= 3)
    {
        return make_shared<op::v4::Interpolate>(
            new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    }
    return make_shared<op::v4::Interpolate>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_attrs);
}

namespace
{
    static constexpr size_t data_port = 0;
    static constexpr size_t target_shape_port = 1;
    static constexpr size_t scales_port = 2;
    static constexpr size_t axes_port = 3;
    static constexpr size_t max_num_of_ports = 4;

    std::vector<int64_t> get_axes_vector(const HostTensorVector& args)
    {
        Shape input_shape{args[data_port]->get_shape()};
        size_t input_rank = input_shape.size();
        size_t num_of_inputs = args.size();

        std::vector<int64_t> axes;

        if (num_of_inputs == max_num_of_ports)
        {
            int64_t* axes_data_ptr = args[axes_port]->get_data_ptr<int64_t>();
            size_t num_of_axes = args[axes_port]->get_shape()[0];
            axes.insert(axes.end(), axes_data_ptr, axes_data_ptr + num_of_axes);
        }
        else
        {
            for (size_t i = 0; i < input_rank; ++i)
            {
                axes.push_back(i);
            }
        }

        return axes;
    }

    std::vector<int64_t> get_target_shape_vector(const HostTensorVector& args, size_t num_of_axes)
    {
        std::vector<int64_t> target_shape;

        int64_t* target_shape_ptr = args[target_shape_port]->get_data_ptr<int64_t>();
        target_shape.insert(target_shape.end(), target_shape_ptr, target_shape_ptr + num_of_axes);

        return target_shape;
    }

    std::vector<float> get_scales_vector(const HostTensorVector& args,
                                         const Shape& input_shape,
                                         const op::v4::Interpolate::InterpolateAttrs& attrs,
                                         std::vector<int64_t> axes)
    {
        using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;

        std::vector<float> scales;
        size_t num_of_axes = axes.size();
        if (attrs.shape_calculation_mode == ShapeCalcMode::scales)
        {
            float* scales_ptr = args[scales_port]->get_data_ptr<float>();
            scales.insert(scales.end(), scales_ptr, scales_ptr + num_of_axes);
        }
        else
        {
            auto target_shape = get_target_shape_vector(args, num_of_axes);
            for (size_t i = 0; i < num_of_axes; ++i)
            {
                size_t axis = axes[i];
                float scale =
                    static_cast<float>(target_shape[i]) / static_cast<float>(input_shape[axis]);
                scales.push_back(scale);
            }
        }
        return scales;
    }

    template <typename T>
    std::vector<T> correct_pad(const std::vector<T>& p, size_t rank)
    {
        size_t pad_len = p.size();
        if (pad_len == rank)
        {
            return p;
        }

        std::vector<T> result;

        if (pad_len > rank)
        {
            result.insert(result.end(), p.begin(), p.begin() + rank);
        }
        else
        {
            result = p;
            result.insert(result.end(), rank - pad_len, T{});
        }

        return result;
    }
}

void op::v4::Interpolate::correct_pads()
{
    PartialShape input_shape = PartialShape(get_input_partial_shape(0));
    if (input_shape.rank().is_dynamic())
    {
        return;
    }
    const auto input_rank = input_shape.rank().get_length();

    m_attrs.pads_begin = correct_pad(m_attrs.pads_begin, input_rank);
    m_attrs.pads_end = correct_pad(m_attrs.pads_end, input_rank);
}

static void pad_input_data(const uint8_t* data_ptr,
                           uint8_t* padded_data_ptr,
                           size_t type_size,
                           const Shape& input_shape,
                           const Shape& padded_input_shape,
                           const std::vector<size_t>& pads_begin)
{
    CoordinateTransform input_transform(input_shape);
    CoordinateTransform padded_transform(padded_input_shape);

    for (const Coordinate& input_coord : input_transform)
    {
        auto padded_coord = input_coord;
        size_t i = 0;
        for (size_t pad : pads_begin)
        {
            padded_coord[i] += pad;
            ++i;
        }
        uint8_t* dst_ptr = padded_data_ptr + type_size * padded_transform.index(padded_coord);
        const uint8_t* src_ptr = data_ptr + type_size * input_transform.index(input_coord);
        memcpy(dst_ptr, src_ptr, type_size);
    }
}

bool op::v4::Interpolate::evaluate(const HostTensorVector& outputs,
                                   const HostTensorVector& inputs) const
{
    element::Type input_et = get_input_element_type(0);
    size_t type_size = input_et.size();

    Shape input_shape{inputs[data_port]->get_shape()};
    Shape padded_input_shape = get_padded_input_shape(input_shape).to_shape();

    auto axes = get_axes_vector(inputs);
    size_t num_of_axes = axes.size();

    auto scales = get_scales_vector(inputs, padded_input_shape, m_attrs, axes);

    PartialShape output_shape{padded_input_shape};

    if (m_attrs.shape_calculation_mode == ShapeCalcMode::scales)
    {
        infer_using_scales(output_shape, axes, scales, padded_input_shape);
    }
    else
    {
        auto sizes = get_target_shape_vector(inputs, num_of_axes);
        infer_using_shapes(output_shape, axes, sizes);
    }

    Shape out_shape = output_shape.to_shape();

    outputs[0]->set_element_type(inputs[0]->get_element_type());
    outputs[0]->set_shape(out_shape);

    size_t bytes_in_padded_input = shape_size(padded_input_shape) * type_size;

    std::vector<uint8_t> padded_input_data(bytes_in_padded_input, 0);

    const uint8_t* data_ptr = inputs[0]->get_data_ptr<uint8_t>();
    uint8_t* padded_data_ptr = padded_input_data.data();

    pad_input_data(
        data_ptr, padded_data_ptr, type_size, input_shape, padded_input_shape, m_attrs.pads_begin);

    switch (input_et)
    {
    case element::Type_t::f32:
        runtime::reference::interpolate<float>(reinterpret_cast<float*>(padded_data_ptr),
                                               padded_input_shape,
                                               scales,
                                               axes,
                                               outputs[0]->get_data_ptr<float>(),
                                               out_shape,
                                               m_attrs);
        break;
    case element::Type_t::f16:
        runtime::reference::interpolate<float16>(reinterpret_cast<float16*>(padded_data_ptr),
                                                 padded_input_shape,
                                                 scales,
                                                 axes,
                                                 outputs[0]->get_data_ptr<float16>(),
                                                 out_shape,
                                                 m_attrs);
        break;
    case element::Type_t::i8:
        runtime::reference::interpolate<int8_t>(reinterpret_cast<int8_t*>(padded_data_ptr),
                                                padded_input_shape,
                                                scales,
                                                axes,
                                                outputs[0]->get_data_ptr<int8_t>(),
                                                out_shape,
                                                m_attrs);
        break;
    default:;
    }

    return true;
}

namespace ngraph
{
    template <>
    NGRAPH_API EnumNames<op::v4::Interpolate::InterpolateMode>&
        EnumNames<op::v4::Interpolate::InterpolateMode>::get()
    {
        static auto enum_names = EnumNames<op::v4::Interpolate::InterpolateMode>(
            "op::v4::Interpolate::InterpolateMode",
            {{"nearest", op::v4::Interpolate::InterpolateMode::nearest},
             {"linear", op::v4::Interpolate::InterpolateMode::linear},
             {"linear_onnx", op::v4::Interpolate::InterpolateMode::linear_onnx},
             {"cubic", op::v4::Interpolate::InterpolateMode::cubic}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v4::Interpolate::InterpolateMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v4::Interpolate::InterpolateMode& type)
    {
        return s << as_string(type);
    }

    template <>
    NGRAPH_API EnumNames<op::v4::Interpolate::ShapeCalcMode>&
        EnumNames<op::v4::Interpolate::ShapeCalcMode>::get()
    {
        static auto enum_names = EnumNames<op::v4::Interpolate::ShapeCalcMode>(
            "op::v4::Interpolate::ShapeCalcMode",
            {{"sizes", op::v4::Interpolate::ShapeCalcMode::sizes},
             {"scales", op::v4::Interpolate::ShapeCalcMode::scales}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v4::Interpolate::ShapeCalcMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v4::Interpolate::ShapeCalcMode& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::v4::Interpolate::CoordinateTransformMode>&
        EnumNames<op::v4::Interpolate::CoordinateTransformMode>::get()
    {
        static auto enum_names = EnumNames<op::v4::Interpolate::CoordinateTransformMode>(
            "op::v4::Interpolate::CoordinateTransformMode",
            {{"half_pixel", op::v4::Interpolate::CoordinateTransformMode::half_pixel},
             {"pytorch_half_pixel",
              op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel},
             {"asymmetric", op::v4::Interpolate::CoordinateTransformMode::asymmetric},
             {"tf_half_pixel_for_nn",
              op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn},
             {"align_corners", op::v4::Interpolate::CoordinateTransformMode::align_corners}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<op::v4::Interpolate::CoordinateTransformMode>::type_info;

    std::ostream& operator<<(std::ostream& s,
                             const op::v4::Interpolate::CoordinateTransformMode& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::v4::Interpolate::NearestMode>& EnumNames<op::v4::Interpolate::NearestMode>::get()
    {
        static auto enum_names = EnumNames<op::v4::Interpolate::NearestMode>(
            "op::v4::Interpolate::NearestMode",
            {{"round_prefer_floor", op::v4::Interpolate::NearestMode::round_prefer_floor},
             {"round_prefer_ceil", op::v4::Interpolate::NearestMode::round_prefer_ceil},
             {"floor", op::v4::Interpolate::NearestMode::floor},
             {"ceil", op::v4::Interpolate::NearestMode::ceil},
             {"simple", op::v4::Interpolate::NearestMode::simple}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v4::Interpolate::NearestMode>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v4::Interpolate::NearestMode& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph
