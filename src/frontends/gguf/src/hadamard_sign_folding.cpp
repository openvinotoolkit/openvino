// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hadamard_sign_folding.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <utility>

#include "openvino/core/model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::frontend::gguf {
namespace {
using Constant = ov::op::v0::Constant;
using Multiply = ov::op::v1::Multiply;
using Squeeze = ov::op::v0::Squeeze;
using Reshape = ov::op::v1::Reshape;
constexpr char sign_marker[] = "gguf_hadamard_input_sign";

bool is_sign_multiply(const std::shared_ptr<Multiply>& multiply) {
    if (!multiply)
        return false;
    const auto& info = multiply->get_rt_info();
    const auto marker = info.find(sign_marker);
    return marker != info.end() && marker->second.is<bool>() && marker->second.as<bool>();
}

std::shared_ptr<Constant> exact_constant(const ov::Output<ov::Node>& value) {
    const auto node = value.get_node_shared_ptr();
    if (const auto constant = ov::as_type_ptr<Constant>(node))
        return constant;
    if (!ov::is_type<ov::op::v0::Convert>(node) && !ov::is_type<ov::op::v1::ConvertLike>(node))
        return nullptr;
    const auto source = exact_constant(node->input_value(0));
    if (!source || value.get_element_type().is_dynamic())
        return nullptr;
    const auto convert = std::make_shared<ov::op::v0::Convert>(source, value.get_element_type());
    ov::TensorVector converted{ov::Tensor(value.get_element_type(), source->get_shape())};
    if (!convert->evaluate(converted, {source->get_tensor_view()}))
        return nullptr;
    const auto result = std::make_shared<Constant>(converted.front());
    const auto reverse = std::make_shared<ov::op::v0::Convert>(result, source->get_element_type());
    ov::TensorVector restored{ov::Tensor(source->get_element_type(), source->get_shape())};
    if (!reverse->evaluate(restored, converted) ||
        std::memcmp(source->get_data_ptr(), restored.front().data(), source->get_byte_size()) != 0)
        return nullptr;
    return result;
}

bool supported_float(const ov::element::Type& type) {
    return type == ov::element::f16 || type == ov::element::bf16 || type == ov::element::f32 ||
           type == ov::element::f64;
}

void collect_constant_sources(const ov::Output<ov::Node>& value, ov::NodeVector& sources) {
    const auto node = value.get_node_shared_ptr();
    sources.push_back(node);
    if (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::ConvertLike>(node))
        collect_constant_sources(node->input_value(0), sources);
}

bool preserves_shape(const ov::PartialShape& data,
                     const ov::Shape& constant,
                     const ov::op::AutoBroadcastSpec& broadcast) {
    if (broadcast.m_type == ov::op::AutoBroadcastType::NONE)
        return data.is_static() && data.to_shape() == constant;
    if (broadcast.m_type != ov::op::AutoBroadcastType::NUMPY || data.rank().is_dynamic() ||
        constant.size() > static_cast<size_t>(data.rank().get_length()))
        return false;
    const auto offset = data.size() - constant.size();
    for (size_t axis = 0; axis < constant.size(); ++axis) {
        if (constant[axis] != 1 && (!data[offset + axis].is_static() ||
                                    data[offset + axis].get_length() != static_cast<int64_t>(constant[axis])))
            return false;
    }
    return true;
}

std::shared_ptr<Constant> signed_scale(const std::shared_ptr<Constant>& scale, const std::shared_ptr<Constant>& signs) {
    ov::PartialShape shape = scale->get_shape();
    if (!ov::PartialShape::broadcast_merge_into(shape, signs->get_shape(), ov::op::AutoBroadcastType::NUMPY) ||
        ov::shape_size(shape.to_shape()) == 0)
        return nullptr;
    const auto values = scale->cast_vector<double>();
    const auto sign_values = signs->cast_vector<double>();
    for (const auto value : values)
        if (!std::isfinite(value))
            return nullptr;
    for (const auto sign : sign_values)
        if (sign != 1.0 && sign != -1.0)
            return nullptr;
    // Multiplying a finite value by +-1 is exact, including for f16/bf16 evaluated through f32.
    ov::TensorVector folded{ov::Tensor(scale->get_element_type(), shape.to_shape())};
    if (!std::make_shared<Multiply>(scale, signs)
             ->evaluate(folded, {scale->get_tensor_view(), signs->get_tensor_view()}))
        return nullptr;
    return std::make_shared<Constant>(folded.front());
}

bool merge_info(const ov::NodeVector& sources, const std::shared_ptr<ov::Node>& target) {
    ov::Node::RTMap plain;
    for (const auto& source : sources) {
        for (const auto& [key, value] : source->get_rt_info()) {
            if (key == sign_marker || key == "opset" || value.is<ov::RuntimeAttribute>())
                continue;
            const auto found = plain.find(key);
            if (found != plain.end() && found->second != value)
                return false;
            plain[key] = value;
        }
    }
    ov::copy_runtime_info(sources, target);
    for (const auto& [key, value] : plain)
        target->get_rt_info()[key] = value;
    target->get_rt_info().erase(sign_marker);
    return true;
}

bool merge_output_info(const ov::OutputVector& sources, ov::Output<ov::Node> target) {
    ov::Node::RTMap plain;
    for (const auto& source : sources) {
        for (const auto& [key, value] : source.get_rt_info()) {
            if (value.is<ov::RuntimeAttribute>())
                continue;
            const auto found = plain.find(key);
            if (found != plain.end() && found->second != value)
                return false;
            plain[key] = value;
        }
    }
    ov::copy_output_runtime_info(sources, {target});
    for (const auto& [key, value] : plain)
        target.get_rt_info()[key] = value;
    return true;
}

std::shared_ptr<ov::Node> clone_branch_node(const std::shared_ptr<ov::Node>& source, const ov::OutputVector& inputs) {
    const auto result = source->clone_with_new_inputs(inputs);
    result->get_rt_info() = source->get_rt_info();
    result->set_friendly_name(source->get_friendly_name());
    result->output(0).get_rt_info() = source->output(0).get_rt_info();
    result->output(0).set_names(source->output(0).get_names());
    return result;
}

bool has_channels(const ov::PartialShape& shape, size_t channels) {
    return shape.rank().is_static() && shape.size() >= 2 && shape[shape.size() - 1].is_static() &&
           shape[shape.size() - 1].get_length() == static_cast<int64_t>(channels);
}

void redirect_shape_queries(const std::shared_ptr<ov::Node>& source, const std::shared_ptr<ov::Node>& replacement) {
    for (auto input : source->output(0).get_target_inputs())
        if (input.get_index() == 0 &&
            (ov::is_type<ov::op::v0::ShapeOf>(input.get_node()) || ov::is_type<ov::op::v3::ShapeOf>(input.get_node())))
            input.replace_source_output(replacement->output(0));
}

bool fold_up_projection(const std::shared_ptr<Multiply>& rotation) {
    std::shared_ptr<Constant> signs;
    size_t sign_port = 0;
    for (; sign_port < 2; ++sign_port) {
        signs = exact_constant(rotation->input_value(sign_port));
        if (signs)
            break;
    }
    if (!signs || !supported_float(signs->get_element_type()) || signs->get_shape().empty())
        return false;
    const auto& sign_shape = signs->get_shape();
    const auto channels = sign_shape.back();
    if (channels == 0 || ov::shape_size(sign_shape) != channels)
        return false;
    const auto sign_values = signs->cast_vector<double>();
    if (std::any_of(sign_values.begin(), sign_values.end(), [](double value) {
            return value != 1.0 && value != -1.0;
        }))
        return false;
    auto data = rotation->input_value(1 - sign_port);
    if (data.get_element_type() != signs->get_element_type() || !has_channels(data.get_partial_shape(), channels) ||
        !preserves_shape(data.get_partial_shape(), sign_shape, rotation->get_autob()))
        return false;
    ov::NodeVector views;
    while (ov::is_type<Reshape>(data.get_node()) || ov::is_type<Squeeze>(data.get_node()) ||
           ov::is_type<ov::op::v0::Unsqueeze>(data.get_node())) {
        const auto view = data.get_node_shared_ptr();
        if (view->get_input_size() != 2 || !has_channels(view->get_input_partial_shape(0), channels) ||
            !has_channels(view->get_output_partial_shape(0), channels))
            return false;
        views.push_back(view);
        data = view->input_value(0);
    }
    const auto swiglu = ov::as_type_ptr<Multiply>(data.get_node_shared_ptr());
    if (!swiglu || is_sign_multiply(swiglu))
        return false;
    size_t up_port = 0;
    std::shared_ptr<ov::op::v0::MatMul> up;
    for (; up_port < 2; ++up_port) {
        up = ov::as_type_ptr<ov::op::v0::MatMul>(swiglu->get_input_node_shared_ptr(up_port));
        if (up && ov::is_type<ov::op::v4::Swish>(swiglu->get_input_node_shared_ptr(1 - up_port)))
            break;
    }
    if (up_port == 2 || !up->get_transpose_b() || !has_channels(up->get_output_partial_shape(0), channels) ||
        up->get_output_partial_shape(0) != swiglu->get_input_partial_shape(1 - up_port) ||
        up->get_output_partial_shape(0) != swiglu->get_output_partial_shape(0))
        return false;

    auto weights = up->input_value(1);
    std::shared_ptr<ov::op::v0::Convert> output_convert;
    if ((output_convert = ov::as_type_ptr<ov::op::v0::Convert>(weights.get_node_shared_ptr()))) {
        const auto type = output_convert->get_destination_type();
        if (output_convert->get_input_element_type(0) != ov::element::f16 ||
            (type != ov::element::f16 && type != ov::element::f32 && type != ov::element::f64))
            return false;
        weights = output_convert->input_value(0);
    }
    const auto weight_view = ov::as_type_ptr<Reshape>(weights.get_node_shared_ptr());
    if (!weight_view || !weights.get_partial_shape().is_static() || weights.get_shape().size() != 2 ||
        weights.get_shape()[0] != channels || weights.get_element_type() != ov::element::f16)
        return false;
    const auto scaled = ov::as_type_ptr<Multiply>(weight_view->get_input_node_shared_ptr(0));
    if (!scaled)
        return false;
    size_t scale_port = 0;
    std::shared_ptr<Constant> scale;
    for (; scale_port < 2; ++scale_port) {
        scale = exact_constant(scaled->input_value(scale_port));
        if (scale)
            break;
    }
    if (!scale || scale->get_element_type() != ov::element::f16)
        return false;
    const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(scaled->get_input_node_shared_ptr(1 - scale_port));
    if (!subtract)
        return false;
    const auto code_convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(0));
    const auto zero_point = exact_constant(subtract->input_value(1));
    if (!code_convert || code_convert->get_destination_type() != ov::element::f16 || !zero_point ||
        zero_point->get_element_type() != ov::element::f16)
        return false;
    const auto codes = ov::as_type_ptr<Constant>(code_convert->get_input_node_shared_ptr(0));
    if (!codes || codes->get_element_type() != ov::element::u2 || codes->get_shape().size() < 2 ||
        codes->get_shape()[0] != channels || ov::shape_size(codes->get_shape()) == 0 ||
        ov::shape_size(codes->get_shape()) != ov::shape_size(weights.get_shape()) ||
        !preserves_shape(codes->get_shape(), zero_point->get_shape(), subtract->get_autob()) ||
        !preserves_shape(codes->get_shape(), scale->get_shape(), scaled->get_autob()))
        return false;
    for (const auto value : zero_point->cast_vector<double>())
        if (value != 1.0)
            return false;
    for (const auto value : scale->cast_vector<double>())
        if (!std::isfinite(value))
            return false;

    const auto* packed = static_cast<const uint8_t*>(codes->get_data_ptr());
    const auto count = ov::shape_size(codes->get_shape());
    if (std::any_of(packed, packed + count / 4, [](uint8_t value) {
            return (value & (value >> 1) & 0x55) != 0;
        }))
        return false;
    for (size_t i = count / 4 * 4; i < count; ++i)
        if (((packed[i / 4] >> (2 * (i % 4))) & 3) == 3)
            return false;
    ov::Shape row_shape(codes->get_shape().size(), 1);
    row_shape[0] = channels;
    const auto row_signs = Constant::create(ov::element::f16, row_shape, sign_values);
    const auto folded = signed_scale(scale, row_signs);
    if (!folded || !preserves_shape(codes->get_shape(), folded->get_shape(), scaled->get_autob()))
        return false;
    folded->set_friendly_name(scale->get_friendly_name());
    folded->output(0).get_rt_info() = scale->output(0).get_rt_info();
    folded->output(0).set_names(scale->output(0).get_names());
    ov::NodeVector constant_sources;
    collect_constant_sources(scaled->input_value(scale_port), constant_sources);
    collect_constant_sources(rotation->input_value(sign_port), constant_sources);
    if (!merge_info(constant_sources, folded))
        return false;
    auto inputs = scaled->input_values();
    inputs[scale_port] = folded;
    auto replacement = clone_branch_node(scaled, inputs);
    replacement = clone_branch_node(weight_view, {replacement, weight_view->input_value(1)});
    if (output_convert)
        replacement = clone_branch_node(output_convert, {replacement});
    replacement = clone_branch_node(up, {up->input_value(0), replacement});
    std::vector<std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>> shape_replacements;
    shape_replacements.reserve(views.size() + 2);
    shape_replacements.emplace_back(up, replacement);
    inputs = swiglu->input_values();
    inputs[up_port] = replacement;
    replacement = clone_branch_node(swiglu, inputs);
    shape_replacements.emplace_back(swiglu, replacement);
    for (auto view = views.rbegin(); view != views.rend(); ++view) {
        replacement = clone_branch_node(*view, {replacement, (*view)->input_value(1)});
        shape_replacements.emplace_back(*view, replacement);
    }
    const auto original = rotation->input_value(1 - sign_port);
    if (!merge_info({original.get_node_shared_ptr(), rotation}, replacement) ||
        !merge_output_info({original, rotation->output(0)}, replacement->output(0)))
        return false;
    auto names = original.get_names();
    const auto& rotation_names = rotation->output(0).get_names();
    names.insert(rotation_names.begin(), rotation_names.end());
    replacement->output(0).set_names(names);
    rotation->output(0).replace(replacement);
    for (const auto& [source, target] : shape_replacements)
        redirect_shape_queries(source, target);
    return true;
}

bool squeeze_axes(const std::shared_ptr<Squeeze>& squeeze, std::set<size_t>& axes) {
    const auto input_shape = squeeze->get_input_partial_shape(0);
    if (input_shape.rank().is_dynamic() || input_shape.size() == 0 || squeeze->get_input_size() != 2)
        return false;
    const auto constant = exact_constant(squeeze->input_value(1));
    if (!constant || ov::shape_size(constant->get_shape()) == 0)
        return false;
    const auto rank = static_cast<int64_t>(input_shape.size());
    for (auto axis : constant->cast_vector<int64_t>()) {
        if (axis < 0)
            axis += rank;
        if (axis < 0 || axis >= rank - 1 || !input_shape[axis].is_static() || input_shape[axis].get_length() != 1)
            return false;
        axes.insert(static_cast<size_t>(axis));
    }
    ov::PartialShape expected;
    for (size_t axis = 0; axis < input_shape.size(); ++axis)
        if (axes.count(axis) == 0)
            expected.push_back(input_shape[axis]);
    return squeeze->get_output_partial_shape(0) == expected;
}

struct SignView {
    ov::Output<ov::Node> output;
    std::shared_ptr<Squeeze> squeeze;
    std::set<size_t> axes;
    ov::NodeVector rotations;
};

std::shared_ptr<Constant> lift_signs(const std::shared_ptr<Constant>& signs, const SignView& view, size_t source_rank) {
    ov::Shape shape(source_rank, 1);
    const auto& sign_shape = signs->get_shape();
    const auto view_rank = source_rank - view.axes.size();
    size_t view_axis = 0;
    for (size_t axis = 0; axis < source_rank; ++axis) {
        if (view.axes.count(axis) != 0)
            continue;
        if (view_axis >= view_rank - sign_shape.size())
            shape[axis] = sign_shape[view_axis - (view_rank - sign_shape.size())];
        ++view_axis;
    }
    const auto first = std::find_if(shape.begin(), shape.end(), [](size_t dim) {
        return dim != 1;
    });
    shape.erase(shape.begin(), first);
    return std::make_shared<Constant>(signs->get_element_type(), shape, signs->get_data_ptr());
}

void fold_scale(const std::shared_ptr<Multiply>& scaled) {
    std::shared_ptr<Constant> scale;
    size_t scale_port = 0;
    for (; scale_port < 2; ++scale_port) {
        scale = exact_constant(scaled->input_value(scale_port));
        if (scale)
            break;
    }
    if (!scale || !supported_float(scale->get_element_type()))
        return;
    const auto data = scaled->input_value(1 - scale_port);
    if (data.get_element_type() != scale->get_element_type() ||
        !preserves_shape(data.get_partial_shape(), scale->get_shape(), scaled->get_autob()))
        return;

    std::vector<SignView> views{{scaled->output(0), nullptr, {}, {}}};
    for (const auto& consumer : scaled->output(0).get_target_inputs()) {
        const auto squeeze = ov::as_type_ptr<Squeeze>(consumer.get_node()->shared_from_this());
        if (!squeeze)
            continue;
        std::set<size_t> axes;
        if (consumer.get_index() != 0 || !squeeze_axes(squeeze, axes))
            return;
        views.push_back({squeeze->output(0), squeeze, std::move(axes), {}});
    }
    ov::NodeVector constant_sources;
    collect_constant_sources(scaled->input_value(scale_port), constant_sources);
    std::shared_ptr<Constant> signs;
    for (auto& view : views) {
        for (const auto& consumer : view.output.get_target_inputs()) {
            const auto node = consumer.get_node()->shared_from_this();
            if (ov::is_type<ov::op::v0::ShapeOf>(node) || ov::is_type<ov::op::v3::ShapeOf>(node) ||
                (!view.squeeze && ov::is_type<Squeeze>(node)))
                continue;
            const auto rotation = ov::as_type_ptr<Multiply>(node);
            if (!is_sign_multiply(rotation))
                return;
            const auto sign_input = rotation->input_value(1 - consumer.get_index());
            if (sign_input == view.output)
                return;
            const auto constant = exact_constant(sign_input);
            if (!constant || constant->get_element_type() != scale->get_element_type() ||
                !preserves_shape(view.output.get_partial_shape(), constant->get_shape(), rotation->get_autob()))
                return;
            const auto candidate = lift_signs(constant, view, data.get_partial_shape().size());
            if (signs && (signs->get_shape() != candidate->get_shape() ||
                          std::memcmp(signs->get_data_ptr(), candidate->get_data_ptr(), signs->get_byte_size()) != 0))
                return;
            signs = candidate;
            view.rotations.push_back(node);
            collect_constant_sources(sign_input, constant_sources);
        }
    }
    if (!signs)
        return;
    const auto folded = signed_scale(scale, signs);
    if (!folded || !preserves_shape(data.get_partial_shape(), folded->get_shape(), scaled->get_autob()))
        return;
    const auto replacement = std::make_shared<Multiply>(data, folded, scaled->get_autob());
    if (!merge_info(constant_sources, folded))
        return;
    ov::OutputVector replacements;
    for (const auto& view : views) {
        const std::shared_ptr<ov::Node> target =
            view.squeeze ? std::static_pointer_cast<ov::Node>(
                               std::make_shared<Squeeze>(replacement, view.squeeze->input_value(1)))
                         : replacement;
        ov::NodeVector sources{view.output.get_node_shared_ptr()};
        sources.insert(sources.end(), view.rotations.begin(), view.rotations.end());
        ov::OutputVector output_sources{view.output};
        for (const auto& rotation : view.rotations)
            output_sources.push_back(rotation->output(0));
        if (!merge_info(sources, target) || !merge_output_info(output_sources, target->output(0)))
            return;
        target->set_friendly_name(view.output.get_node()->get_friendly_name());
        auto names = view.output.get_names();
        for (const auto& rotation : view.rotations) {
            const auto& output_names = rotation->output(0).get_names();
            names.insert(output_names.begin(), output_names.end());
        }
        target->output(0).set_names(names);
        replacements.push_back(target->output(0));
    }
    for (size_t i = 0; i < views.size(); ++i) {
        for (const auto& rotation : views[i].rotations)
            rotation->output(0).replace(replacements[i]);
        views[i].output.replace(replacements[i]);
    }
}
}  // namespace

void fold_hadamard_input_signs(const std::shared_ptr<ov::Model>& model) {
    std::set<ov::Node*> visited;
    for (const auto& node : model->get_ordered_ops()) {
        const auto rotation = ov::as_type_ptr<Multiply>(node);
        if (!is_sign_multiply(rotation))
            continue;
        if (fold_up_projection(rotation))
            continue;
        for (const auto& input : rotation->input_values()) {
            auto source = input.get_node_shared_ptr();
            if (const auto squeeze = ov::as_type_ptr<Squeeze>(source))
                source = squeeze->input_value(0).get_node_shared_ptr();
            const auto scaled = ov::as_type_ptr<Multiply>(source);
            if (scaled && !is_sign_multiply(scaled) && visited.insert(scaled.get()).second)
                fold_scale(scaled);
        }
    }
}

}  // namespace ov::frontend::gguf
