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
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/reference/interpolate.hpp"

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
} // namespace ngraph

// Interpolate v4

constexpr NodeTypeInfo op::v4::Interpolate::type_info;

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
    visitor.on_attribute("attrs", m_attrs);
    return true;
}

static const char* cannot_define_axes =
    "Cannot define axes of interpolation when there is "
    "only two inputs and input data have dynamic rank";

std::vector<int64_t> op::v4::Interpolate::get_axes() const
{
    auto inputs = input_values();
    if (inputs.size() <= 3)
    {
        PartialShape input_shape = PartialShape(get_input_partial_shape(0));
        if (input_shape.rank().is_dynamic())
        {
            throw std::invalid_argument(cannot_define_axes);
        }
        const auto input_rank = input_shape.rank().get_length();
        std::vector<int64_t> default_value(input_rank);
        for (int64_t i = 0; i < input_rank; ++i)
        {
            default_value[i] = i;
        }
        return default_value;
    }

    std::vector<int64_t> result;
    if (auto axes_node = as_type_ptr<op::Constant>(input_value(3).get_node_shared_ptr()))
    {
        result = axes_node->cast_vector<int64_t>();
    }

    return result;
}

static constexpr float epsilon = 1.0e-6f;

void op::v4::Interpolate::infer_using_scales(PartialShape& output_shape,
                                             const std::vector<int64_t>& axes,
                                             const PartialShape& padded_input_shape)
{
    auto const_scales = as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());
    if (!const_scales)
    {
        return;
    }

    auto scales = const_scales->cast_vector<float>();
    size_t i = 0;
    for (auto axis : axes)
    {
        if (padded_input_shape[axis].is_static())
        {
            float padded_len = static_cast<float>(padded_input_shape[axis].get_length());
            int64_t new_dim = static_cast<int64_t>((padded_len + epsilon) * scales[i]);
            output_shape[axis] = Dimension(new_dim);
        }
        ++i;
    }
}

void op::v4::Interpolate::infer_using_shapes(PartialShape& output_shape,
                                             const std::vector<int64_t>& axes)
{
    auto const_shape = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr());
    if (!const_shape)
    {
        return;
    }

    auto out_shape = const_shape->cast_vector<int64_t>();
    size_t i = 0;
    for (auto axis : axes)
    {
        output_shape[axis] = Dimension(out_shape[i++]);
    }
}

void op::v4::Interpolate::validate_and_infer_types()
{
    PartialShape output_shape = PartialShape(get_input_partial_shape(0));
    PartialShape padded_input_shape = output_shape;

    if (!output_shape.rank().is_static())
    {
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    auto axes = get_axes();
    correct_pads();

    const auto input_rank = output_shape.rank().get_length();

    for (size_t i = 0; i < input_rank; ++i)
    {
        if (output_shape[i].is_static())
        {
            auto new_length =
                m_attrs.pads_begin[i] + m_attrs.pads_end[i] + output_shape[i].get_length();
            output_shape[i] = Dimension(new_length);
            padded_input_shape[i] = Dimension(new_length);
        }
    }

    for (auto axis : axes)
    {
        NGRAPH_CHECK(axis < input_rank);
        output_shape[axis] = Dimension::dynamic();
    }

    if (m_attrs.shape_calculation_mode == ShapeCalcMode::scales)
    {
        infer_using_scales(output_shape, axes, padded_input_shape);
    }
    else
    {
        infer_using_shapes(output_shape, axes);
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
    static constexpr std::size_t data_port = 0;
    static constexpr std::size_t target_shape_port = 1;
    static constexpr std::size_t scales_port = 2;
    static constexpr std::size_t axes_port = 3;
    static constexpr std::size_t max_num_of_ports = 4;

    using size_t_vector = std::vector<std::size_t>;
    using int64_vector = std::vector<int64_t>;

    int64_vector get_axes_vector(const HostTensorVector& args)
    {
        Shape input_shape{args[data_port]->get_shape()};
        std::size_t input_rank = input_shape.size();
        std::size_t num_of_inputs = args.size();

        int64_vector axes;

        if (num_of_inputs == max_num_of_ports)
        {
            int64_t* axes_data_ptr = args[axes_port]->get_data_ptr<int64_t>();
            std::size_t num_of_axes = args[axes_port]->get_shape()[0];
            axes.insert(axes.end(), axes_data_ptr, axes_data_ptr + num_of_axes);
        }
        else
        {
            for (std::size_t i = 0; i < input_rank; ++i)
            {
                axes.push_back(i);
            }
        }

        return axes;
    }

    int64_vector get_target_shape_vector(const HostTensorVector& args, std::size_t num_of_axes)
    {
        int64_vector target_shape;

        int64_t* target_shape_ptr = args[target_shape_port]->get_data_ptr<int64_t>();
        target_shape.insert(target_shape.end(), target_shape_ptr, target_shape_ptr + num_of_axes);

        return target_shape;
    }

    std::vector<float> get_scales_vector(const HostTensorVector& args,
                                         const Shape& input_shape,
                                         const op::v4::Interpolate::InterpolateAttrs& attrs,
                                         int64_vector axes)
    {
        using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;

        std::vector<float> scales;
        std::size_t num_of_axes = axes.size();
        if (attrs.shape_calculation_mode == ShapeCalcMode::scales)
        {
            float* scales_ptr = args[scales_port]->get_data_ptr<float>();
            scales.insert(scales.end(), scales_ptr, scales_ptr + num_of_axes);
        }
        else
        {
            auto target_shape = get_target_shape_vector(args, num_of_axes);
            for (std::size_t i = 0; i < num_of_axes; ++i)
            {
                std::size_t axis = axes[i];
                float scale =
                    static_cast<float>(target_shape[i]) / static_cast<float>(input_shape[axis]);
                scales.push_back(scale);
            }
        }
        return scales;
    }

    size_t_vector get_padded_input_shape(const Shape& input_shape,
                                         const size_t_vector& pads_begin,
                                         const size_t_vector& pads_end)
    {
        size_t_vector result = input_shape;
        std::size_t rank = input_shape.size();
        for (std::size_t i = 0; i < rank; ++i)
        {
            result[i] += pads_begin[i] + pads_end[i];
        }

        return result;
    }

    size_t_vector shape_infer_with_scales(const size_t_vector& padded_shape,
                                          const int64_vector& axes,
                                          const std::vector<float>& scales)
    {
        auto out_shape = padded_shape;
        std::size_t num_of_axes = axes.size();
        for (std::size_t i = 0; i < num_of_axes; ++i)
        {
            int64_t axis = axes[i];
            float scaled_len = static_cast<float>(padded_shape[axis]) * scales[i];
            out_shape[axis] = static_cast<std::size_t>(scaled_len);
        }
        return out_shape;
    }

    size_t_vector shape_infer_with_target_shape(const size_t_vector& padded_shape,
                                                const int64_vector& axes,
                                                const int64_vector& target_shape)
    {
        auto out_shape = padded_shape;
        std::size_t num_of_axes = axes.size();
        for (std::size_t i = 0; i < num_of_axes; ++i)
        {
            out_shape[axes[i]] = target_shape[i];
        }
        return out_shape;
    }

    template <typename T>
    std::vector<T> correct_pad(const std::vector<T>& p, std::size_t rank)
    {
        std::size_t pad_len = p.size();
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

    struct InfoToCallReference
    {
        Shape input_shape;
        Shape padded_input_shape;
        Shape out_shape;
        std::vector<size_t> pads_begin;
        std::vector<size_t> pads_end;
        std::vector<int64_t> axes;
        std::vector<float> scales;
    };

    using InterpolateV4Attrs = op::v4::Interpolate::InterpolateAttrs;

    InfoToCallReference get_info_to_call_reference(const HostTensorVector& args,
                                                   const InterpolateV4Attrs& attrs)
    {
        using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;

        Shape input_shape{args[data_port]->get_shape()};
        std::size_t input_rank = input_shape.size();

        auto axes = get_axes_vector(args);
        std::size_t num_of_axes = axes.size();

        auto scales = get_scales_vector(args, input_shape, attrs, axes);

        auto pads_begin = correct_pad(attrs.pads_begin, input_rank);
        auto pads_end = correct_pad(attrs.pads_end, input_rank);

        auto padded_input_shape_vector = get_padded_input_shape(input_shape, pads_begin, pads_end);
        std::vector<std::size_t> out_shape_vector;

        if (attrs.shape_calculation_mode == ShapeCalcMode::scales)
        {
            out_shape_vector = shape_infer_with_scales(padded_input_shape_vector, axes, scales);
        }
        else
        {
            auto target_shape = get_target_shape_vector(args, num_of_axes);
            out_shape_vector =
                shape_infer_with_target_shape(padded_input_shape_vector, axes, target_shape);
        }

        Shape out_shape{out_shape_vector};
        Shape padded_input_shape{padded_input_shape_vector};

        return {input_shape, padded_input_shape, out_shape, pads_begin, pads_end, axes, scales};
    }

    template <element::Type_t ET>
    inline bool evaluate(const HostTensorVector& args,
                         const HostTensorPtr& out,
                         const op::v4::Interpolate::InterpolateAttrs& attrs)
    {
        using T = typename element_type_traits<ET>::value_type;

        auto info_for_reference = get_info_to_call_reference(args, attrs);

        out->set_element_type(args[0]->get_element_type());
        out->set_shape(info_for_reference.out_shape);

        std::vector<T> padded_input_data(shape_size(info_for_reference.padded_input_shape), T{});

        CoordinateTransform input_transform(info_for_reference.input_shape);
        CoordinateTransform padded_transform(info_for_reference.padded_input_shape);

        const T* data_ptr = args[0]->get_data_ptr<ET>();
        T* padded_data_ptr = padded_input_data.data();

        for (const Coordinate& input_coord : input_transform)
        {
            auto padded_coord = input_coord;
            std::size_t i = 0;
            for (std::size_t pad : info_for_reference.pads_begin)
            {
                padded_coord[i] += pad;
            }
            padded_data_ptr[padded_transform.index(padded_coord)] =
                data_ptr[input_transform.index(input_coord)];
        }

        runtime::reference::interpolate<T>(padded_input_data.data(),
                                           info_for_reference.padded_input_shape,
                                           info_for_reference.scales,
                                           info_for_reference.axes,
                                           out->get_data_ptr<ET>(),
                                           info_for_reference.out_shape,
                                           attrs);
        return true;
    }

    bool evaluate_interpolate_v4(const HostTensorVector& args,
                                 const HostTensorPtr& out,
                                 const op::v4::Interpolate::InterpolateAttrs& attrs)
    {
        bool rc = true;
        switch (args[0]->get_element_type())
        {
            TYPE_CASE(i8)(args, out, attrs);
            break;
            TYPE_CASE(f16)(args, out, attrs);
            break;
            TYPE_CASE(f32)(args, out, attrs);
            break;
        default: rc = false; break;
        }
        return rc;
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

bool op::v4::Interpolate::evaluate(const HostTensorVector& outputs,
                                   const HostTensorVector& inputs) const
{
    return evaluate_interpolate_v4(inputs, outputs[0], m_attrs);
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

    constexpr DiscreteTypeInfo AttributeAdapter<op::v4::Interpolate::InterpolateAttrs>::type_info;

    AttributeAdapter<op::v4::Interpolate::InterpolateAttrs>::AttributeAdapter(
        op::v4::Interpolate::InterpolateAttrs& ref)
        : m_ref(ref)
    {
    }

    bool AttributeAdapter<op::v4::Interpolate::InterpolateAttrs>::visit_attributes(
        AttributeVisitor& visitor)
    {
        visitor.on_attribute("mode", m_ref.mode);
        visitor.on_attribute("shape_calculation_mode", m_ref.shape_calculation_mode);
        visitor.on_attribute("coordinate_transformation_mode",
                             m_ref.coordinate_transformation_mode);
        visitor.on_attribute("nearest_mode", m_ref.nearest_mode);
        visitor.on_attribute("antialias", m_ref.antialias);
        visitor.on_attribute("pads_begin", m_ref.pads_begin);
        visitor.on_attribute("pads_end", m_ref.pads_end);
        visitor.on_attribute("cube_coeff", m_ref.cube_coeff);
        return true;
    }
} // namespace ngraph
