// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/util/log.hpp"
#include "pt_framework_node.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using namespace ov::op;

OutputVector NodeContext::as_constant() const {
    auto dtype = m_decoder->get_output_type(0);
    if (dtype.is<type::Str>()) {
        // Cannot represent string as Constant, creating FrameworkNode
        const auto& str = m_decoder->as_string();
        auto fw_node = std::make_shared<PtFrameworkNode>(m_decoder, OutputVector{});
        auto attrs = fw_node->get_attrs();
        attrs["string_value"] = str;
        attrs[PtFrameworkNode::failed_conversion_key] =
            "String constant cannot be converted to OpenVINO opset and should be removed by consuming operation.";
        fw_node->set_attrs(attrs);
        return {fw_node};
    } else if (dtype.is<type::PyNone>()) {
        // Cannot represent None as Constant, creating FrameworkNode
        auto fw_node = std::make_shared<PtFrameworkNode>(m_decoder, OutputVector{});
        auto attrs = fw_node->get_attrs();
        attrs["none_value"] = "";
        attrs[PtFrameworkNode::failed_conversion_key] =
            "None constant cannot be converted to OpenVINO opset and should be removed by consuming operation.";
        fw_node->set_attrs(attrs);
        return {fw_node};
    } else {
        auto c_outs = m_decoder->as_constant();
        FRONT_END_OP_CONVERSION_CHECK(c_outs.size() == 1, "Constant must have exactly one output.");
        return c_outs;
    }
}

std::shared_ptr<Node> NodeContext::mark_node(std::shared_ptr<Node> ov_node) const {
    ov_node = m_decoder->mark_node(ov_node);
    return ov_node;
}

void NodeContext::mutate_input(size_t index, Output<Node> ov_output) const {
    FRONT_END_GENERAL_CHECK(!input_is_none(index), "Input is none with index: ", index);
    auto input_id = m_decoder_inputs.at(index);
    FRONT_END_GENERAL_CHECK(m_tensor_map->count(input_id), "No tensor corresponding input: ", input_id, " exist.");
    m_translate_session->encode_tensor_name(ov_output, input_id, {m_decoder->get_input_debug_name(index)});
    (*m_tensor_map)[input_id] = ov_output;
    m_mutated_tensors->insert(input_id);

    // Resolve aliases
    auto back_input_id = input_id;
    auto back_node_input = ov_output;
    while (m_translate_session->m_may_be_alias.count(back_input_id)) {
        // Create node to reverseprop data. While loop is needed for the cases when alias to tensor point to another
        // alias to tensor. In that case we need to create a chain of reverseprop ops
        size_t in_tensor;
        std::shared_ptr<TorchDecoder> node;
        Output<Node> node_converted_output;
        std::tie(in_tensor, node, node_converted_output) = m_translate_session->m_may_be_alias.at(back_input_id);
        auto reverseprop_node = m_translate_session->get_reverseprop_op(node, node_converted_output, back_node_input);
        if (m_tensor_map->count(in_tensor)) {
            // Tensor is not found in the scope of this body, need to get it from internal context and mark mutated
            OPENVINO_DEBUG << "Couldn't find in the current body the initial aliased tensor: " << in_tensor
                           << " for operation: " << node->get_op_type() << " creating new body input.";
            get_tensor_from_model_or_create_input(in_tensor);
        }
        m_translate_session->encode_tensor_name(reverseprop_node, in_tensor);
        (*m_tensor_map)[in_tensor] = reverseprop_node;
        m_mutated_tensors->insert(in_tensor);
        OPENVINO_DEBUG << "Propagated back data from tensor: " << back_input_id << " to tensor: " << in_tensor << ".\n";
        back_input_id = in_tensor;
        back_node_input = reverseprop_node;
    }
}

void NodeContext::add_tensor_to_context(size_t index, Output<Node> ov_output) const {
    if (m_tensor_map->count(index)) {
        OPENVINO_DEBUG << "[ WARNING ] Current context has tensor " << index << ". Assuming mutated output.\n";
    }
    m_translate_session->encode_tensor_name(ov_output, index);
    (*m_tensor_map)[index] = ov_output;
}

Output<Node> NodeContext::get_tensor_from_model_or_create_input(size_t index) const {
    if (m_tensor_map->find(index) != m_tensor_map->end()) {
        return m_tensor_map->at(index);
    } else {
        // nested subgraphs case
        auto parameter = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
        m_translate_session->encode_tensor_name(parameter->output(0), index);
        (*m_tensor_map)[index] = parameter;
        m_external_parameters->push_back(parameter);
        OPENVINO_DEBUG << "Nested case, created: " << parameter << '\n';
        return parameter;
    }
}

Output<Node> NodeContext::get_input_from_visible_context(size_t index) const {
    FRONT_END_GENERAL_CHECK(index < get_input_size(), "Index ", index, " is lower then number of inputs.");
    auto input_tensor = get_input(static_cast<int>(index));
    auto input_node = input_tensor.get_node_shared_ptr();
    if (std::dynamic_pointer_cast<v0::Parameter>(input_node)) {
        // We need to look into external context for inputs that would be feed into this parameter
        size_t tensor_idx = m_translate_session->decode_tensor_name(input_node->output(0));
        if (m_ext_tensor_map.count(tensor_idx)) {
            input_tensor = m_ext_tensor_map.at(tensor_idx);
        }
    }
    return input_tensor;
}

std::shared_ptr<ov::Model> NodeContext::convert_subgraph(size_t index) const {
    auto subgraph_decoder = m_decoder->get_subgraph_decoder(index);

    // Extend external context with internal tensors except Parameter nodes, because internal Parameters are created to
    // link internal context with external
    TensorMap ext_map(m_ext_tensor_map);
    // map::insert does not update elements if their key is already in map; so if we have real tensors in outter scope
    // we will not add Parameters we created in inner scope.
    ext_map.insert(m_tensor_map->begin(), m_tensor_map->end());

    auto model = m_translate_session->convert_pytorch_model(subgraph_decoder, ext_map);
    // Remove unused parameters, they could be created as inputs to the parts of graph that weren't
    // used for generating output.
    for (auto i = subgraph_decoder->inputs().size(); i < model->get_parameters().size(); i++) {
        auto parameter = model->get_parameters()[i];
        if (parameter->output(0).get_target_inputs().empty()) {
            // There is no consumers: safe to remove
            OPENVINO_DEBUG << "Removing parameter " << parameter
                           << " in converted Pytorch model, because it is never used\n";
            model->remove_parameter(parameter);
        }
    }
    return model;
}

bool NodeContext::input_is_none(size_t index) const {
    bool res = index >= m_inputs_is_none.size() || m_inputs_is_none.at(index);
    if (!res) {
        // check case when input is from outside body
        auto input = get_input_from_visible_context(index);
        res = is_none_node(input);
    }
    return res;
}

namespace {
std::shared_ptr<v0::Constant> get_constant_at_input(const NodeContext& ctx, size_t index, bool allow_empty = true) {
    FRONT_END_GENERAL_CHECK(!ctx.input_is_none(index), "Input with index: ", index, " is none.");
    auto input_val = ctx.get_input_from_visible_context(index);
    if (ctx.get_input_type(index).is<type::List>()) {
        if (allow_empty && is_empty_list(input_val))
            return {};
        input_val = concat_list_construct(input_val);
    }
    auto constant = ov::util::get_constant_from_source(input_val);
    FRONT_END_GENERAL_CHECK(constant, "Input with index ", index, " cannot be interpreted as Constant: ", input_val);
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
std::string NodeContext::const_input<std::string>(size_t index) const {
    FRONT_END_GENERAL_CHECK(!input_is_none(index), "Input with index: ", index, " is none.");
    auto input_node = get_input_from_visible_context(index).get_node_shared_ptr();
    auto input = std::dynamic_pointer_cast<PtFrameworkNode>(input_node);
    FRONT_END_GENERAL_CHECK(input,
                            "Input node with index ",
                            index,
                            " cannot be interpreted as FrameworkNode with string constant: ",
                            input_node);
    return input->get_decoder()->as_string();
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
    if (input_is_none(index))
        return {};
    auto input_val = get_input_from_visible_context(index);
    if (auto input = std::dynamic_pointer_cast<PtFrameworkNode>(input_val.get_node_shared_ptr())) {
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

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
