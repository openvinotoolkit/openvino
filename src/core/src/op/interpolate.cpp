// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "interpolate_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

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

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes, make_tensor_accessor());
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> op::v0::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Interpolate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v0::Interpolate>(new_args.at(0), new_args.at(1), m_attrs);
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

void op::v0::Interpolate::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
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

void ov::op::v4::Interpolate::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Interpolate_validate_and_infer_types);

    InterpolateBase::validate_and_infer_types();

    validate_sizes_element_type(get_input_element_type(1));
    validate_scales_element_type(get_input_element_type(2));

    if (input_values().size() == 4) {
        validate_axes_element_type(get_input_element_type(3));
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

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes =
        shape_infer(this, input_shapes, m_attrs.pads_begin, m_attrs.pads_end, make_tensor_accessor());
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<ov::Node> ov::op::v4::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Interpolate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() <= 3) {
        return std::make_shared<ov::op::v4::Interpolate>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    }
    return std::make_shared<ov::op::v4::Interpolate>(new_args.at(0),
                                                     new_args.at(1),
                                                     new_args.at(2),
                                                     new_args.at(3),
                                                     m_attrs);
}

namespace {
static constexpr size_t data_port = 0;
static constexpr size_t target_shape_port = 1;
size_t get_scales_port(const ov::op::util::InterpolateBase* node) {
    constexpr size_t scales_port = 2;
    if (ov::is_type<ov::op::v11::Interpolate>(node)) {
        return scales_port - 1;
    }
    return scales_port;
}

size_t get_axes_port(const ov::op::util::InterpolateBase* node) {
    constexpr size_t axes_port = 3;
    if (ov::is_type<ov::op::v11::Interpolate>(node)) {
        return axes_port - 1;
    }
    return axes_port;
}

size_t get_max_num_of_ports(const ov::op::util::InterpolateBase* node) {
    constexpr size_t max_num_of_ports = 4;
    if (ov::is_type<ov::op::v11::Interpolate>(node)) {
        return max_num_of_ports - 1;
    }
    return max_num_of_ports;
}

std::vector<float> get_scales_vector(const ov::op::util::InterpolateBase* node,
                                     const ov::TensorVector& args,
                                     const ov::Shape& input_shape,
                                     const ov::op::util::InterpolateBase::InterpolateAttrs& attrs,
                                     const std::vector<int64_t>& axes) {
    using scales_t = float;
    constexpr auto f32_cast = ov::util::Cast<scales_t>();

    if (attrs.shape_calculation_mode == ov::op::util::InterpolateBase::ShapeCalcMode::SCALES) {
        return ov::get_tensor_data_as<scales_t>(args[get_scales_port(node)], f32_cast);
    } else {
        auto scales = ov::get_tensor_data_as<scales_t>(args[target_shape_port], f32_cast);
        auto scales_iter = scales.begin();
        for (const auto axis : axes) {
            *scales_iter /= input_shape[axis];
            ++scales_iter;
        }
        return scales;
    }
}

bool evaluate_interpolate(const ov::op::util::InterpolateBase* node,
                          const std::vector<ov::PartialShape>& input_shapes,
                          const ov::Shape& out_shape,
                          std::vector<size_t>& pads_begin,
                          std::vector<size_t>& pads_end,
                          const ov::ITensorAccessor& ta,
                          ov::TensorVector& outputs,
                          const ov::TensorVector& inputs) {
    outputs[0].set_shape(out_shape);
    auto padded_input_shape =
        ov::op::interpolate::make_padded_shape(input_shapes.front(), pads_begin.begin(), pads_end.begin()).to_shape();
    const auto has_axes_input = (input_shapes.size() == get_max_num_of_ports(node));
    const auto axes = ov::op::interpolate::get_axes<ov::PartialShape>(node,
                                                                      get_axes_port(node),
                                                                      has_axes_input,
                                                                      out_shape.size(),
                                                                      ta);
    const auto& m_attrs = node->get_attrs();
    const auto scales = get_scales_vector(node, inputs, padded_input_shape, m_attrs, *axes);

    const auto input_et = inputs[0].get_element_type();
    const auto type_size = input_et.size();
    const auto bytes_in_padded_input = ov::shape_size(padded_input_shape) * type_size;
    auto padded_input_data = std::vector<uint8_t>(bytes_in_padded_input, 0);

    auto* data_ptr = static_cast<const uint8_t*>(inputs[data_port].data());
    auto* padded_data_ptr = padded_input_data.data();

    ov::reference::pad_input_data(data_ptr,
                                  padded_data_ptr,
                                  type_size,
                                  inputs[data_port].get_shape(),
                                  padded_input_shape,
                                  pads_begin);
    using ov::element::Type_t;
    switch (input_et) {
    case Type_t::f32:
        ov::reference::interpolate<float>(reinterpret_cast<float*>(padded_data_ptr),
                                          padded_input_shape,
                                          scales,
                                          *axes,
                                          outputs[0].data<float>(),
                                          out_shape,
                                          m_attrs);
        break;
    case Type_t::f16:
    case Type_t::bf16:
        return ov::util::evaluate_node_with_unsupported_precision(node, outputs, inputs);
    case Type_t::i8:
        ov::reference::interpolate<int8_t>(reinterpret_cast<int8_t*>(padded_data_ptr),
                                           padded_input_shape,
                                           scales,
                                           *axes,
                                           outputs[0].data<int8_t>(),
                                           out_shape,
                                           m_attrs);
        break;
    case Type_t::u8:
        ov::reference::interpolate<uint8_t>(reinterpret_cast<uint8_t*>(padded_data_ptr),
                                            padded_input_shape,
                                            scales,
                                            *axes,
                                            outputs[0].data<uint8_t>(),
                                            out_shape,
                                            m_attrs);
        break;
    default:
        return false;
    }

    return true;
}
}  // namespace

bool ov::op::v4::Interpolate::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v4_Interpolate_evaluate);
    auto pads_begin = m_attrs.pads_begin;
    auto pads_end = m_attrs.pads_end;
    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto ta = make_tensor_accessor(inputs);
    const auto out_shape = shape_infer(this, input_shapes, pads_begin, pads_end, ta).front().to_shape();
    return evaluate_interpolate(this, input_shapes, out_shape, pads_begin, pads_end, ta, outputs, inputs);
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
        return false;
    }
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

std::shared_ptr<Node> op::v11::Interpolate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v11_Interpolate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<op::v11::Interpolate>(new_args.at(0), new_args.at(1), m_attrs);
    }
    return std::make_shared<op::v11::Interpolate>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
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

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes =
        shape_infer(this, input_shapes, m_attrs.pads_begin, m_attrs.pads_end, make_tensor_accessor());
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

AttributeAdapter<op::v0::Interpolate::InterpolateMode>::~AttributeAdapter() = default;

bool ov::op::v11::Interpolate::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v11_Interpolate_evaluate);

    auto pads_begin = m_attrs.pads_begin;
    auto pads_end = m_attrs.pads_end;
    const auto ta = make_tensor_accessor(inputs);
    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto out_shape = shape_infer(this, input_shapes, pads_begin, pads_end, ta).front().to_shape();
    return evaluate_interpolate(this, input_shapes, out_shape, pads_begin, pads_end, ta, outputs, inputs);
}

bool ov::op::v11::Interpolate::has_evaluate() const {
    OV_OP_SCOPE(v11_Interpolate_has_evaluate);
    switch (get_input_element_type(0)) {
    case ov::element::i8:
    case ov::element::u8:
    case ov::element::bf16:
    case ov::element::f16:
    case ov::element::f32:
        return true;
    default:
        return false;
    }
}

}  // namespace ov
