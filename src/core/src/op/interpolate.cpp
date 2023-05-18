// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <interpolate_shape_inference.hpp>
#include <ngraph/validation_util.hpp>
#include <numeric>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/reference/interpolate.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
namespace ov {
ov::op::v0::Interpolate::Interpolate(const Output<Node>& image,
                                     const Output<Node>& output_shape,
                                     const Attributes& attrs)
    : Op({image, output_shape}),
      m_attrs(attrs) {
    ov::mark_as_precision_sensitive(input(1));
    constructor_validate_and_infer_types();
}

bool ov::op::v0::Interpolate::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Interpolate_visit_attributes);
    visitor.on_attribute("align_corners", m_attrs.align_corners);
    visitor.on_attribute("antialias", m_attrs.antialias);
    visitor.on_attribute("axes", m_attrs.axes);
    visitor.on_attribute("mode", m_attrs.mode);
    visitor.on_attribute("pads_begin", m_attrs.pads_begin);
    visitor.on_attribute("pads_end", m_attrs.pads_end);
    return true;
}

void ov::op::v0::Interpolate::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Interpolate_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "output shape must be an integral number.");
    set_input_is_relevant_to_shape(1);

    const auto& input_shape = get_input_partial_shape(0);
    const auto& target_spatial_shape = get_input_partial_shape(1);
    std::vector<ov::PartialShape> input_shapes = {input_shape, target_spatial_shape};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};

    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

shared_ptr<Node> op::v0::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Interpolate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Interpolate>(new_args.at(0), new_args.at(1), m_attrs);
}

std::ostream& operator<<(std::ostream& s, const op::v0::Interpolate::InterpolateMode& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<ov::op::v0::Interpolate::InterpolateMode>&
EnumNames<ov::op::v0::Interpolate::InterpolateMode>::get() {
    static auto enum_names = EnumNames<ov::op::v0::Interpolate::InterpolateMode>(
        "op::v0::Interpolate::InterpolateMode",
        {{"nearest", ov::op::v0::Interpolate::InterpolateMode::NEAREST},
         {"linear", ov::op::v0::Interpolate::InterpolateMode::LINEAR},
         {"cubic", ov::op::v0::Interpolate::InterpolateMode::CUBIC},
         {"area", ov::op::v0::Interpolate::InterpolateMode::AREA}});
    return enum_names;
}
}  // namespace ov

// Interpolate v4
ov::op::v4::Interpolate::Interpolate(const Output<Node>& image,
                                     const Output<Node>& output_shape,
                                     const Output<Node>& scales,
                                     const Output<Node>& axes,
                                     const ov::op::v4::Interpolate::InterpolateAttrs& attrs)
    : util::InterpolateBase{image, output_shape, scales, axes, attrs} {
    constructor_validate_and_infer_types();
}

ov::op::v4::Interpolate::Interpolate(const Output<Node>& image,
                                     const Output<Node>& output_shape,
                                     const Output<Node>& scales,
                                     const ov::op::v4::Interpolate::InterpolateAttrs& attrs)
    : util::InterpolateBase{image, output_shape, scales, attrs} {
    ov::mark_as_precision_sensitive(input(2));
    constructor_validate_and_infer_types();
}

std::vector<int64_t> ov::op::v4::Interpolate::get_axes() const {
    auto inputs = input_values();
    if (inputs.size() <= 3) {
        ov::PartialShape input_shape = ov::PartialShape(get_input_partial_shape(0));
        NODE_VALIDATION_CHECK(this,
                              input_shape.rank().is_static(),
                              "Could not define axes of interpolation because there are "
                              "only three inputs and input data has a dynamic rank.");

        const auto input_rank = input_shape.rank().get_length();
        std::vector<int64_t> default_value(input_rank);
        std::iota(default_value.begin(), default_value.end(), 0);

        return default_value;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto axes_node = get_constant_from_source(input_value(3));
    OPENVINO_SUPPRESS_DEPRECATED_END
    NODE_VALIDATION_CHECK(this, axes_node, "Input 'axes' should be Constant or foldable.");

    return axes_node->cast_vector<int64_t>();
}

static constexpr float epsilon = 1.0e-6f;

void ov::op::v4::Interpolate::infer_using_scales(ov::PartialShape& output_shape,
                                                 const std::vector<int64_t>& axes,
                                                 const std::vector<float>& scales,
                                                 const ov::PartialShape& padded_input_shape) const {
    size_t i = 0;
    for (auto axis : axes) {
        const auto& current_dim = padded_input_shape[axis];
        float multiplier = scales[i] + epsilon;

        int64_t new_lower_bound = util::multiply_bound_and_scale(current_dim.get_min_length(), multiplier);
        int64_t new_upper_bound = util::multiply_bound_and_scale(current_dim.get_max_length(), multiplier);

        output_shape[axis] = Dimension(new_lower_bound, new_upper_bound);
        ++i;
    }
}

void ov::op::v4::Interpolate::infer_using_shapes(ov::PartialShape& output_shape,
                                                 const std::vector<int64_t>& axes,
                                                 const std::vector<int64_t>& sizes) const {
    size_t i = 0;
    for (auto axis : axes) {
        output_shape[axis] = Dimension(sizes[i++]);
    }
}

ov::PartialShape ov::op::v4::Interpolate::get_padded_input_shape(const ov::PartialShape& input_shape) const {
    const auto input_rank = input_shape.rank().get_length();

    ov::PartialShape padded_input_shape = input_shape;

    for (int64_t i = 0; i < input_rank; ++i) {
        if (input_shape[i].is_static()) {
            auto new_length = m_attrs.pads_begin[i] + m_attrs.pads_end[i] + input_shape[i].get_length();
            padded_input_shape[i] = Dimension(new_length);
        }
    }

    return padded_input_shape;
}

void ov::op::v4::Interpolate::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Interpolate_validate_and_infer_types);

    InterpolateBase::validate_and_infer_types();

    validate_sizes_element_type(get_input_element_type(1));

    validate_scales_element_type(get_input_element_type(2));

    if (input_values().size() == 4) {
        validate_axes_element_type(get_input_element_type(3));
    }

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes;
    const auto& input_shape = get_input_partial_shape(0);
    const auto& target_spatial_shape = get_input_partial_shape(1);
    const auto& scales = get_input_partial_shape(2);
    if (input_values().size() == 3) {
        input_shapes = {input_shape, target_spatial_shape, scales};
    } else {
        const auto& axes = get_input_partial_shape(3);
        input_shapes = {input_shape, target_spatial_shape, scales, axes};
    }

    const auto interpolation_mode_check = [](const op::util::InterpolateBase::InterpolateMode mode) {
        constexpr std::array<op::util::InterpolateBase::InterpolateMode, 4> allowed_modes = {
            op::util::InterpolateBase::InterpolateMode::NEAREST,
            op::util::InterpolateBase::InterpolateMode::LINEAR,
            op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX,
            op::util::InterpolateBase::InterpolateMode::CUBIC};

        return std::find(std::begin(allowed_modes), std::end(allowed_modes), mode) != std::end(allowed_modes);
    };

    NODE_VALIDATION_CHECK(this,
                          interpolation_mode_check(m_attrs.mode),
                          "Unsupported interpolation mode used with version 4 of the Interpolate op: ",
                          as_string(m_attrs.mode));

    util::correct_pads_attr(this, m_attrs.pads_begin, m_attrs.pads_end, input_shapes);
    shape_infer(this, m_attrs.pads_begin, m_attrs.pads_end, input_shapes, output_shapes, {});
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

shared_ptr<ov::Node> ov::op::v4::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Interpolate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() <= 3) {
        return make_shared<ov::op::v4::Interpolate>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    }
    return make_shared<ov::op::v4::Interpolate>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                m_attrs);
}

namespace {
static constexpr size_t data_port = 0;
static constexpr size_t target_shape_port = 1;
static constexpr size_t scales_port = 2;
static constexpr size_t axes_port = 3;
static constexpr size_t max_num_of_ports = 4;

std::vector<int64_t> get_axes_vector(const ngraph::HostTensorVector& args) {
    ov::Shape input_shape{args[data_port]->get_shape()};
    size_t input_rank = input_shape.size();
    size_t num_of_inputs = args.size();

    std::vector<int64_t> axes;

    if (num_of_inputs == max_num_of_ports) {
        auto axes_arg = args[axes_port];
        size_t num_of_axes = args[axes_port]->get_shape()[0];
        axes.reserve(num_of_axes);

        if (axes_arg->get_element_type() == ov::element::i64) {
            int64_t* axes_ptr = axes_arg->get_data_ptr<int64_t>();
            axes.insert(axes.end(), axes_ptr, axes_ptr + num_of_axes);
        } else if (axes_arg->get_element_type() == ov::element::i32) {
            int32_t* axes_ptr = axes_arg->get_data_ptr<int32_t>();
            for (size_t i = 0; i < num_of_axes; ++i)
                axes.push_back(axes_ptr[i]);
        } else {
            OPENVINO_ASSERT(false, "Failed to process ", axes_arg->get_element_type());
        }
    } else {
        for (size_t i = 0; i < input_rank; ++i) {
            axes.push_back(i);
        }
    }

    return axes;
}

std::vector<int64_t> get_target_shape_vector(const ngraph::HostTensorVector& args, size_t num_of_axes) {
    std::vector<int64_t> target_shape;
    target_shape.reserve(num_of_axes);

    auto target_shape_arg = args[target_shape_port];
    if (target_shape_arg->get_element_type() == ov::element::i64) {
        int64_t* target_shape_ptr = target_shape_arg->get_data_ptr<int64_t>();
        target_shape.insert(target_shape.end(), target_shape_ptr, target_shape_ptr + num_of_axes);
    } else if (target_shape_arg->get_element_type() == ov::element::i32) {
        int32_t* target_shape_ptr = target_shape_arg->get_data_ptr<int32_t>();
        for (size_t i = 0; i < num_of_axes; ++i)
            target_shape.push_back(target_shape_ptr[i]);
    } else {
        OPENVINO_ASSERT(false, "Failed to process ", target_shape_arg->get_element_type());
    }

    return target_shape;
}

std::vector<float> get_scales_vector(const ngraph::HostTensorVector& args,
                                     const ov::Shape& input_shape,
                                     const ov::op::v4::Interpolate::InterpolateAttrs& attrs,
                                     std::vector<int64_t> axes) {
    std::vector<float> scales;
    size_t num_of_axes = axes.size();
    if (attrs.shape_calculation_mode == ov::op::util::InterpolateBase::ShapeCalcMode::SCALES) {
        float* scales_ptr = args[scales_port]->get_data_ptr<float>();
        scales.insert(scales.end(), scales_ptr, scales_ptr + num_of_axes);
    } else {
        auto target_shape = get_target_shape_vector(args, num_of_axes);
        for (size_t i = 0; i < num_of_axes; ++i) {
            size_t axis = axes[i];
            float scale = static_cast<float>(target_shape[i]) / static_cast<float>(input_shape[axis]);
            scales.push_back(scale);
        }
    }
    return scales;
}

template <typename T>
std::vector<T> correct_pad(const std::vector<T>& p, size_t rank) {
    size_t pad_len = p.size();
    if (pad_len == rank) {
        return p;
    }

    std::vector<T> result;

    if (pad_len > rank) {
        result.insert(result.end(), p.begin(), p.begin() + rank);
    } else {
        result = p;
        result.insert(result.end(), rank - pad_len, T{});
    }

    return result;
}
}  // namespace

void ov::op::v4::Interpolate::correct_pads() {
    ov::PartialShape input_shape = ov::PartialShape(get_input_partial_shape(0));
    if (input_shape.rank().is_dynamic()) {
        return;
    }
    const auto input_rank = input_shape.rank().get_length();

    m_attrs.pads_begin = correct_pad(m_attrs.pads_begin, input_rank);
    m_attrs.pads_end = correct_pad(m_attrs.pads_end, input_rank);
}

static void pad_input_data(const uint8_t* data_ptr,
                           uint8_t* padded_data_ptr,
                           size_t type_size,
                           const ov::Shape& input_shape,
                           const ov::Shape& padded_input_shape,
                           const std::vector<size_t>& pads_begin) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    ngraph::CoordinateTransform input_transform(input_shape);
    ngraph::CoordinateTransform padded_transform(padded_input_shape);

    for (const ngraph::Coordinate& input_coord : input_transform) {
        auto padded_coord = input_coord;
        size_t i = 0;
        for (size_t pad : pads_begin) {
            padded_coord[i] += pad;
            ++i;
        }
        uint8_t* dst_ptr = padded_data_ptr + type_size * padded_transform.index(padded_coord);
        const uint8_t* src_ptr = data_ptr + type_size * input_transform.index(input_coord);
        memcpy(dst_ptr, src_ptr, type_size);
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

bool ov::op::v4::Interpolate::evaluate_interpolate(const HostTensorVector& outputs,
                                                   const HostTensorVector& inputs) const {
    element::Type input_et = get_input_element_type(0);
    size_t type_size = input_et.size();

    ov::Shape input_shape{inputs[data_port]->get_shape()};
    ov::Shape padded_input_shape = get_padded_input_shape(input_shape).to_shape();

    auto axes = get_axes_vector(inputs);
    size_t num_of_axes = axes.size();

    auto scales = get_scales_vector(inputs, padded_input_shape, m_attrs, axes);

    ov::PartialShape output_shape{padded_input_shape};

    if (m_attrs.shape_calculation_mode == ShapeCalcMode::SCALES) {
        infer_using_scales(output_shape, axes, scales, padded_input_shape);
    } else {
        auto sizes = get_target_shape_vector(inputs, num_of_axes);
        infer_using_shapes(output_shape, axes, sizes);
    }

    ov::Shape out_shape = output_shape.to_shape();

    outputs[0]->set_element_type(inputs[0]->get_element_type());
    outputs[0]->set_shape(out_shape);

    size_t bytes_in_padded_input = shape_size(padded_input_shape) * type_size;

    std::vector<uint8_t> padded_input_data(bytes_in_padded_input, 0);

    const uint8_t* data_ptr = inputs[0]->get_data_ptr<uint8_t>();
    uint8_t* padded_data_ptr = padded_input_data.data();

    pad_input_data(data_ptr, padded_data_ptr, type_size, input_shape, padded_input_shape, m_attrs.pads_begin);

    switch (input_et) {
    case element::Type_t::f32:
        ngraph::runtime::reference::interpolate<float>(reinterpret_cast<float*>(padded_data_ptr),
                                                       padded_input_shape,
                                                       scales,
                                                       axes,
                                                       outputs[0]->get_data_ptr<float>(),
                                                       out_shape,
                                                       m_attrs);
        break;
    case element::Type_t::f16:
        ngraph::runtime::reference::interpolate<float16>(reinterpret_cast<float16*>(padded_data_ptr),
                                                         padded_input_shape,
                                                         scales,
                                                         axes,
                                                         outputs[0]->get_data_ptr<float16>(),
                                                         out_shape,
                                                         m_attrs);
        break;
    case element::Type_t::bf16:
        ngraph::runtime::reference::interpolate<bfloat16>(reinterpret_cast<bfloat16*>(padded_data_ptr),
                                                          padded_input_shape,
                                                          scales,
                                                          axes,
                                                          outputs[0]->get_data_ptr<bfloat16>(),
                                                          out_shape,
                                                          m_attrs);
        break;
    case element::Type_t::i8:
        ngraph::runtime::reference::interpolate<int8_t>(reinterpret_cast<int8_t*>(padded_data_ptr),
                                                        padded_input_shape,
                                                        scales,
                                                        axes,
                                                        outputs[0]->get_data_ptr<int8_t>(),
                                                        out_shape,
                                                        m_attrs);
        break;
    case element::Type_t::u8:
        ngraph::runtime::reference::interpolate<uint8_t>(reinterpret_cast<uint8_t*>(padded_data_ptr),
                                                         padded_input_shape,
                                                         scales,
                                                         axes,
                                                         outputs[0]->get_data_ptr<uint8_t>(),
                                                         out_shape,
                                                         m_attrs);
        break;
    default:;
    }

    return true;
}

bool ov::op::v4::Interpolate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v4_Interpolate_evaluate);
    return evaluate_interpolate(outputs, inputs);
}

bool ov::op::v4::Interpolate::has_evaluate() const {
    OV_OP_SCOPE(v4_Interpolate_has_evaluate);
    switch (get_input_element_type(0)) {
    case ov::element::i8:
    case ov::element::u8:
    case ov::element::bf16:
    case ov::element::f16:
    case ov::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

namespace ov {
op::v11::Interpolate::Interpolate(const Output<Node>& image,
                                  const Output<Node>& scales_or_sizes,
                                  const InterpolateAttrs& attrs)
    : op::util::InterpolateBase{image, scales_or_sizes, attrs} {
    constructor_validate_and_infer_types();
}

op::v11::Interpolate::Interpolate(const Output<Node>& image,
                                  const Output<Node>& scales_or_sizes,
                                  const Output<Node>& axes,
                                  const InterpolateAttrs& attrs)
    : op::util::InterpolateBase{image, scales_or_sizes, axes, attrs} {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v11::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v11_Interpolate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return make_shared<op::v11::Interpolate>(new_args.at(0), new_args.at(1), m_attrs);
    }
    return make_shared<op::v11::Interpolate>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}

void op::v11::Interpolate::validate_and_infer_types() {
    OV_OP_SCOPE(v11_Interpolate_validate_and_infer_types);

    InterpolateBase::validate_and_infer_types();

    const auto& scales_or_sizes_et = get_input_element_type(1);
    if (m_attrs.shape_calculation_mode == ShapeCalcMode::SCALES) {
        validate_scales_element_type(scales_or_sizes_et);
    } else {
        validate_sizes_element_type(scales_or_sizes_et);
    }

    if (input_values().size() == 3) {
        validate_axes_element_type(get_input_element_type(2));
    }

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes;
    const auto& input_shape = get_input_partial_shape(0);
    const auto& scales_or_sizes = get_input_partial_shape(1);
    if (input_values().size() == 2) {
        input_shapes = {input_shape, scales_or_sizes};
    } else {
        const auto& axes = get_input_partial_shape(2);
        input_shapes = {input_shape, scales_or_sizes, axes};
    }

    util::correct_pads_attr(this, m_attrs.pads_begin, m_attrs.pads_end, input_shapes);
    shape_infer(this, m_attrs.pads_begin, m_attrs.pads_end, input_shapes, output_shapes, {});
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}
}  // namespace ov
