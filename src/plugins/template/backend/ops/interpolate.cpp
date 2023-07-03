// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/interpolate.hpp"

#include "evaluate_node.hpp"
#include "interpolate_shape_inference.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::Interpolate>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    ngraph::element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case ngraph::element::Type_t::f64:
        ngraph::runtime::reference::interpolate<double>(inputs[0]->get_data_ptr<double>(),
                                                        op->get_input_partial_shape(0),
                                                        outputs[0]->get_data_ptr<double>(),
                                                        op->get_output_shape(0),
                                                        op->get_attrs());
        break;
    case ngraph::element::Type_t::f32:
        ngraph::runtime::reference::interpolate<float>(inputs[0]->get_data_ptr<float>(),
                                                       op->get_input_partial_shape(0),
                                                       outputs[0]->get_data_ptr<float>(),
                                                       op->get_output_shape(0),
                                                       op->get_attrs());
        break;
    case ngraph::element::Type_t::f16:
        ngraph::runtime::reference::interpolate<ngraph::float16>(inputs[0]->get_data_ptr<ngraph::float16>(),
                                                                 op->get_input_partial_shape(0),
                                                                 outputs[0]->get_data_ptr<ngraph::float16>(),
                                                                 op->get_output_shape(0),
                                                                 op->get_attrs());
        break;
    case ngraph::element::Type_t::bf16:
        ngraph::runtime::reference::interpolate<ngraph::bfloat16>(inputs[0]->get_data_ptr<ngraph::bfloat16>(),
                                                                  op->get_input_partial_shape(0),
                                                                  outputs[0]->get_data_ptr<ngraph::bfloat16>(),
                                                                  op->get_output_shape(0),
                                                                  op->get_attrs());
        break;
    default:;
    }
    return true;
}

template <>
bool evaluate_node<ngraph::op::v0::Interpolate>(std::shared_ptr<ngraph::Node> node,
                                                const ngraph::HostTensorVector& outputs,
                                                const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v0::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

namespace eval {
namespace interpolate {
// The helpers below are similar to the internal utils used in evaluate method of v4::Intepolate core op
// Those functions can be unified and moved to a common place
std::vector<int64_t> get_axes_vector(const ngraph::HostTensorVector& args,
                                     size_t default_size,
                                     size_t axes_port,
                                     size_t max_num_of_ports) {
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
        for (size_t i = 0; i < default_size; ++i) {
            axes.push_back(i);
        }
    }

    return axes;
}

std::vector<int64_t> get_target_shape_vector(const ngraph::HostTensorVector& args,
                                             size_t num_of_axes,
                                             size_t target_shape_port = 1) {
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
                                     const ov::op::util::InterpolateBase::InterpolateAttrs& attrs,
                                     std::vector<int64_t> axes,
                                     size_t scales_port) {
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

namespace v11 {
bool evaluate_interpolate(const std::shared_ptr<ov::op::v11::Interpolate>& op,
                          const ngraph::HostTensorVector& outputs,
                          const ngraph::HostTensorVector& inputs) {
    using namespace ov;

    constexpr size_t data_port = 0;
    constexpr size_t scales_sizes_port = 1;
    constexpr size_t axes_port = 2;
    constexpr size_t max_num_of_ports = 3;

    element::Type input_et = inputs[0]->get_element_type();
    size_t type_size = input_et.size();

    PartialShape input_shape{inputs[data_port]->get_shape()};
    auto m_attrs = op->get_attrs();

    const auto ta = make_tensor_accessor(inputs);
    auto input_shapes = std::vector<PartialShape>();
    std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(input_shapes), [](const HostTensorPtr& ht) {
        return ht->get_shape();
    });
    const auto output_shape =
        ov::op::v11::shape_infer(op.get(), input_shapes, m_attrs.pads_begin, m_attrs.pads_end, ta).front();

    Shape padded_input_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        padded_input_shape.emplace_back(m_attrs.pads_begin[i] + m_attrs.pads_end[i] + input_shape[i].get_length());
    }

    auto axes = get_axes_vector(inputs, inputs[1]->get_shape()[0], axes_port, max_num_of_ports);
    auto scales = get_scales_vector(inputs, padded_input_shape, m_attrs, axes, scales_sizes_port);

    Shape out_shape = output_shape.to_shape();
    outputs[0]->set_shape(out_shape);
    outputs[0]->set_element_type(input_et);

    size_t bytes_in_padded_input = shape_size(padded_input_shape) * type_size;
    std::vector<uint8_t> padded_input_data(bytes_in_padded_input, 0);

    const uint8_t* data_ptr = inputs[0]->get_data_ptr<uint8_t>();
    uint8_t* padded_data_ptr = padded_input_data.data();

    pad_input_data(data_ptr,
                   padded_data_ptr,
                   type_size,
                   input_shape.to_shape(),
                   padded_input_shape,
                   m_attrs.pads_begin);

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
    case element::Type_t::bf16:
        ngraph::runtime::reference::interpolate<bfloat16>(reinterpret_cast<bfloat16*>(padded_data_ptr),
                                                          padded_input_shape,
                                                          scales,
                                                          axes,
                                                          outputs[0]->get_data_ptr<bfloat16>(),
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
    case element::Type_t::u8:
        ngraph::runtime::reference::interpolate<uint8_t>(reinterpret_cast<uint8_t*>(padded_data_ptr),
                                                         padded_input_shape,
                                                         scales,
                                                         axes,
                                                         outputs[0]->get_data_ptr<uint8_t>(),
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
    case element::Type_t::i32:
        ngraph::runtime::reference::interpolate<int32_t>(reinterpret_cast<int32_t*>(padded_data_ptr),
                                                         padded_input_shape,
                                                         scales,
                                                         axes,
                                                         outputs[0]->get_data_ptr<int32_t>(),
                                                         out_shape,
                                                         m_attrs);
        break;
    default:;
    }

    return true;
}
}  // namespace v11
}  // namespace interpolate
}  // namespace eval

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v11::Interpolate>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    return eval::interpolate::v11::evaluate_interpolate(op, outputs, inputs);
}

template <>
bool evaluate_node<ngraph::op::v11::Interpolate>(std::shared_ptr<ngraph::Node> node,
                                                 const ngraph::HostTensorVector& outputs,
                                                 const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v11::Interpolate>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
