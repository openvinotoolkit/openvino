// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"

#include <memory>
#include <string>
#include <vector>

#include "jax_framework_node.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/jax/decoder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/util/log.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {

using namespace ov::op;

OutputVector NodeContext::as_constant() const {
    auto dtype = m_decoder->get_output_type(0);
    if (dtype.is<type::Str>()) {
        OPENVINO_THROW("Dtype string is not supported in JAX yet.");
    } else if (dtype.is<type::PyNone>()) {
        // Cannot represent None as Constant, creating FrameworkNode
        auto fw_node = std::make_shared<JaxFrameworkNode>(m_decoder, OutputVector{});
        auto attrs = fw_node->get_attrs();
        attrs["none_value"] = "";
        attrs[JaxFrameworkNode::failed_conversion_key] =
            "None constant cannot be converted to OpenVINO opset and should be "
            "removed by consuming operation.";
        fw_node->set_attrs(attrs);
        return {fw_node};
    } else {
        auto c_outs = m_decoder->as_constant();
        FRONT_END_OP_CONVERSION_CHECK(c_outs.size() == 1, "Constant must have exactly one output.");
        return c_outs;
    }
}

namespace {
std::shared_ptr<v0::Constant> get_constant_at_input(const NodeContext& ctx, size_t index, bool allow_empty = true) {
    auto input_val = ctx.get_input(static_cast<int>(index));
    if (ctx.get_input_type(index).is<type::List>()) {
        FRONT_END_THROW("Taking list as constant has not been supported in JAX frontend yet.");
    }
    auto constant = ov::util::get_constant_from_source(input_val);
    FRONT_END_GENERAL_CHECK(constant, "Input with index ", index, " cannot be interpreted as Constant: ", input_val);
    return constant;
}

std::shared_ptr<v0::Constant> get_constant_from_params(const NodeContext& ctx, const std::string& name) {
    auto param_val = ctx.get_param(name);
    auto constant = ov::util::get_constant_from_source(param_val);
    FRONT_END_GENERAL_CHECK(constant, "Param with name ", name, " cannot be interpreted as Constant: ", param_val);
    return constant;
}
}  // namespace

template <>
std::vector<int64_t> NodeContext::const_input<std::vector<int64_t>>(size_t index) const {
    auto c = get_constant_at_input(*this, index);
    if (c)
        return c->cast_vector<int64_t>();
    else
        return {};
}

template <>
Strides NodeContext::const_input<Strides>(size_t index) const {
    auto c = get_constant_at_input(*this, index);
    if (c)
        return c->cast_vector<Strides::value_type>();
    else
        return {};
}

template <>
CoordinateDiff NodeContext::const_input<CoordinateDiff>(size_t index) const {
    auto c = get_constant_at_input(*this, index);
    if (c)
        return c->cast_vector<CoordinateDiff::value_type>();
    else
        return {};
}

template <>
Shape NodeContext::const_input<Shape>(size_t index) const {
    auto c = get_constant_at_input(*this, index);
    if (c)
        return c->cast_vector<Shape::value_type>();
    else
        return {};
}

template <>
int32_t NodeContext::const_input<int32_t>(size_t index) const {
    return get_constant_at_input(*this, index, false)->cast_vector<int32_t>()[0];
}

template <>
int64_t NodeContext::const_input<int64_t>(size_t index) const {
    return get_constant_at_input(*this, index, false)->cast_vector<int64_t>()[0];
}

template <>
bool NodeContext::const_input<bool>(size_t index) const {
    return get_constant_at_input(*this, index, false)->cast_vector<bool>()[0];
}

template <>
double NodeContext::const_input<double>(size_t index) const {
    return get_constant_at_input(*this, index, false)->cast_vector<double>()[0];
}

template <>
float NodeContext::const_input<float>(size_t index) const {
    return get_constant_at_input(*this, index, false)->cast_vector<float>()[0];
}

template <>
int64_t NodeContext::const_named_param<int64_t>(const std::string& name) const {
    return get_constant_from_params(*this, name)->cast_vector<int64_t>()[0];
}

template <>
std::vector<int64_t> NodeContext::const_named_param<std::vector<int64_t>>(const std::string& name) const {
    return get_constant_from_params(*this, name)->cast_vector<int64_t>();
}

template <>
std::vector<std::vector<int64_t>> NodeContext::const_named_param<std::vector<std::vector<int64_t>>>(
    const std::string& name) const {
    auto c = get_constant_from_params(*this, name);
    auto shape = c->get_shape();
    auto flatten_vector = c->cast_vector<int64_t>();
    FRONT_END_GENERAL_CHECK(shape.size() == 2, "Expected 2D vector, got ", shape.size(), "D vector.");
    std::vector<std::vector<int64_t>> result(shape[0]);
    for (size_t i = 0; i < shape[0]; ++i) {
        result[i].resize(shape[1]);
        for (size_t j = 0; j < shape[1]; ++j) {
            result[i][j] = flatten_vector[i * shape[1] + j];
        }
    }
    return result;
    // return get_constant_from_params(*this, name)->cast_vector<std::vector<int64_t>>();
}

template <>
Strides NodeContext::const_named_param<Strides>(const std::string& name) const {
    auto c = get_constant_from_params(*this, name);
    if (c)
        return c->cast_vector<Strides::value_type>();
    else
        return {};
}

template <>
CoordinateDiff NodeContext::const_named_param<CoordinateDiff>(const std::string& name) const {
    auto c = get_constant_from_params(*this, name);
    if (c)
        return c->cast_vector<CoordinateDiff::value_type>();
    else
        return {};
}

template <>
std::shared_ptr<v0::Constant> NodeContext::const_named_param<std::shared_ptr<v0::Constant>>(
    const std::string& name) const {
    auto c = get_constant_from_params(*this, name);
    return c;
}

namespace {
template <typename T>
Any get_constant_data(const std::shared_ptr<v0::Constant>& constant) {
    const T* ptr = reinterpret_cast<const T*>(constant->get_data_ptr());
    const auto& shape = constant->get_shape();
    if (is_scalar(shape)) {
        return ptr[0];
    }
    return std::vector<T>(ptr, ptr + shape_size(shape));
}
}  // namespace

Any NodeContext::get_values_from_const_input(int index) const {
    FRONT_END_GENERAL_CHECK(static_cast<size_t>(index) < get_input_size(),
                            "Input with index: ",
                            index,
                            " does not exist.");
    auto input_val = get_input(index);
    if (auto input = ov::as_type_ptr<JaxFrameworkNode>(input_val.get_node_shared_ptr())) {
        const auto& attrs = input->get_attrs();
        if (attrs.find("none_value") != attrs.end()) {
            return {};
        }
        auto it = attrs.find("string_value");
        if (it != attrs.end()) {
            return it->second;
        }
    }
    auto constant = get_constant_at_input(*this, index);
    if (constant) {
        switch (constant->get_element_type()) {
        case element::f32:
            return get_constant_data<float>(constant);
        case element::f64:
            return get_constant_data<double>(constant);
        case element::i32:
            return get_constant_data<int32_t>(constant);
        case element::u32:
            return get_constant_data<uint32_t>(constant);
        case element::i64:
            return get_constant_data<int64_t>(constant);
        case element::u64:
            return get_constant_data<uint64_t>(constant);
        case element::i8:
            return get_constant_data<int8_t>(constant);
        case element::u8:
            return get_constant_data<uint8_t>(constant);
        case element::i16:
            return get_constant_data<int16_t>(constant);
        case element::u16:
            return get_constant_data<uint16_t>(constant);
        case element::boolean:
            return get_constant_data<bool>(constant);
        default:
            FRONT_END_GENERAL_CHECK(false, "Input with index: ", index, " has unsupported type.");
        }
    }
    FRONT_END_GENERAL_CHECK(false, "Input node with index ", index, " cannot be interpreted as constant", input_val);

    return 0;
}

}  // namespace jax
}  // namespace frontend
}  // namespace ov
