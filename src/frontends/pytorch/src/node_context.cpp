// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"

#include "helper_ops/internal_op.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/util/log.hpp"
#include "pt_framework_node.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using namespace ov::op;

namespace {
Output<Node> rebase_view(const Output<Node>& view, const Output<Node>& old_base, const Output<Node>& new_base) {
    std::map<Output<Node>, Output<Node>> replacements{{old_base, new_base}};
    if (ov::is_type<SequenceMark>(old_base.get_node_shared_ptr())) {
        const auto old_elements = old_base.get_node_shared_ptr()->input_values();
        const auto new_elements = new_base.get_node_shared_ptr()->input_values();
        FRONT_END_GENERAL_CHECK(old_elements.size() == new_elements.size(), "An aliased list cannot change length.");
        for (size_t i = 0; i < old_elements.size(); ++i) {
            replacements[old_elements[i]] = new_elements[i];
        }
    }
    if (const auto old_complex = ov::as_type_ptr<ComplexTypeMark>(old_base.get_node_shared_ptr())) {
        const auto new_complex = ov::as_type_ptr<ComplexTypeMark>(new_base.get_node_shared_ptr());
        FRONT_END_GENERAL_CHECK(new_complex, "An alias update must preserve complex element type.");
        replacements[old_complex->get_data()] = new_complex->get_data();
        replacements[old_complex->get_real()] = new_complex->get_real();
        replacements[old_complex->get_imag()] = new_complex->get_imag();
    }
    std::function<Output<Node>(const Output<Node>&)> clone = [&](const Output<Node>& output) -> Output<Node> {
        const auto found = replacements.find(output);
        if (found != replacements.end()) {
            return found->second;
        }
        const auto node = output.get_node_shared_ptr();
        const auto inputs = node->input_values();
        OutputVector new_inputs;
        for (const auto& input : inputs) {
            new_inputs.push_back(clone(input));
        }
        if (new_inputs == inputs) {
            replacements[output] = output;
            return output;
        }
        const auto updated = node->clone_with_new_inputs(new_inputs);
        updated->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node, updated);
        for (size_t i = 0; i < node->get_output_size(); ++i) {
            replacements[node->output(i)] = updated->output(i);
        }
        return updated->output(output.get_index());
    };
    return clone(view);
}
}  // namespace

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
        if (simplified_type_interpret(dtype).is<type::Complex>()) {
            // Add complex mark to complex constant
            c_outs = {mark_node(std::make_shared<ComplexTypeMark>(c_outs[0], c_outs[0].get_element_type()))};
        }
        return c_outs;
    }
}

std::shared_ptr<Node> NodeContext::mark_node(std::shared_ptr<Node> ov_node) const {
    ov_node = m_decoder->mark_node(ov_node);
    return ov_node;
}

void NodeContext::mutate_input(size_t index, Output<Node> ov_output) const {
    FRONT_END_GENERAL_CHECK(!input_is_none(index), "Input is none with index: ", index);
    mutate_tensor(m_decoder_inputs.at(index), ov_output, m_decoder->get_input_debug_name(index));
}

void NodeContext::mutate_input(const std::string& name, Output<Node> ov_output) const {
    mutate_tensor(m_decoder->get_named_input(name), ov_output, name);
}

void NodeContext::mutate_tensor(size_t input_id, Output<Node> ov_output, const std::string& name) const {
    auto tensor_it = m_tensor_map->find(input_id);
    FRONT_END_GENERAL_CHECK(tensor_it != m_tensor_map->end(), "No tensor corresponding input: ", input_id, " exist.");
    const auto previous_value = tensor_it->second;
    m_translate_session->encode_tensor_name(ov_output, input_id, {name});
    tensor_it->second = ov_output;
    m_mutated_tensors->insert(input_id);

    const auto op_type = m_decoder->get_op_type();
    if (op_type.find("aten.unsqueeze_.") == 0 || op_type.find("aten.squeeze_.") == 0 ||
        op_type.find("aten.transpose_.") == 0 || op_type.find("aten.t_.") == 0) {
        // Existing views retain their shape when the base tensor's metadata changes.
        const auto old_shape_view =
            m_translate_session->get_reverseprop_op(m_decoder, ov_output, ov_output, previous_value);
        for (auto& [alias_id, info] : m_translate_session->m_may_be_alias) {
            if (info.base_id == input_id) {
                info.output = rebase_view(info.output, info.base_value, old_shape_view);
                info.base_value = ov_output;
                (*m_tensor_map)[alias_id] = info.output;
            }
        }
        const auto alias = m_translate_session->m_may_be_alias.find(input_id);
        if (alias != m_translate_session->m_may_be_alias.end()) {
            alias->second.output = ov_output;
        }
        return;
    }

    // Resolve aliases
    auto& back_input_id = input_id;
    auto& back_node_input = ov_output;
    while (m_translate_session->m_may_be_alias.count(back_input_id)) {
        // Create node to aliased data. While loop is needed for the cases when alias to tensor point to another
        // alias to tensor. In that case we need to create a chain of reverseprop ops
        auto& alias_info = m_translate_session->m_may_be_alias.at(back_input_id);
        const auto in_tensor = alias_info.base_id;
        auto reverseprop_node = m_translate_session->get_reverseprop_op(alias_info.decoder,
                                                                        alias_info.output,
                                                                        back_node_input,
                                                                        alias_info.base_value);
        m_translate_session->encode_tensor_name(reverseprop_node, in_tensor);
        (*m_tensor_map)[in_tensor] = reverseprop_node;
        m_mutated_tensors->insert(in_tensor);
        OPENVINO_DEBUG("Propagated back data from tensor:", back_input_id, " to tensor: ", in_tensor, "\n");
        back_input_id = in_tensor;
        back_node_input = reverseprop_node;
    }
}

void NodeContext::add_tensor_to_context(size_t index, const Output<Node>& ov_output) const {
#ifdef ENABLE_OPENVINO_DEBUG
    if (m_tensor_map->count(index)) {
        OPENVINO_DEBUG("[ WARNING ] Current context has tensor ", index, ". Assuming mutated output.\n");
    }
#endif
    m_translate_session->encode_tensor_name(ov_output, index);
    (*m_tensor_map)[index] = ov_output;
}

Output<Node> NodeContext::get_tensor_from_model_or_create_input(size_t index) const {
    auto tensor_it = m_tensor_map->find(index);
    if (tensor_it != m_tensor_map->end()) {
        return tensor_it->second;
    } else {
        // nested subgraphs case
        auto parameter = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
        m_translate_session->encode_tensor_name(parameter->output(0), index);
        (*m_tensor_map)[index] = parameter;
        m_external_parameters->push_back(parameter);
        OPENVINO_DEBUG("Nested case, created: ", parameter, "\n");
        return parameter;
    }
}

Output<Node> NodeContext::get_input_from_visible_context(size_t index) const {
    FRONT_END_GENERAL_CHECK(index < get_input_size(), "Index ", index, " is lower than number of inputs.");
    auto input_tensor = get_input(static_cast<int>(index));
    auto input_node = input_tensor.get_node_shared_ptr();
    if (ov::as_type_ptr<v0::Parameter>(input_node)) {
        // We need to look into external context for inputs that would be fed into this parameter
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
    // map::insert does not update elements if their key is already in map; so if we have real tensors in outer scope
    // we will not add Parameters we created in inner scope.
    ext_map.insert(m_tensor_map->begin(), m_tensor_map->end());

    auto model = m_translate_session->convert_pytorch_model(subgraph_decoder, ext_map);
    // Remove unused parameters, they could be created as inputs to the parts of graph that weren't
    // used for generating output.
    for (auto i = subgraph_decoder->inputs().size(); i < model->get_parameters().size(); i++) {
        auto parameter = model->get_parameters()[i];
        if (parameter->output(0).get_target_inputs().empty()) {
            // There is no consumers: safe to remove
            OPENVINO_DEBUG("Removing parameter ", parameter, " in converted Pytorch model, because it is never used\n");
            model->remove_parameter(parameter);
        }
    }
    return model;
}

Output<Node> NodeContext::get_input(int index) const {
    size_t index_ = static_cast<size_t>(index);
    auto input = m_decoder_inputs.at(index);
    if (input == 0) {
        // Case when input can be inlined (possible only for fx decoder)
        if (m_decoder->is_input_inlined(index_)) {
            if (m_decoder->input_is_none(index_)) {
                // some operations like aten.index.Tensor can have None inputs
                auto dummy_decoder = std::make_shared<InternalOpDecoder>("torch::None", 1);
                auto fw_node = std::make_shared<PtFrameworkNode>(dummy_decoder, OutputVector{});
                auto attrs = fw_node->get_attrs();
                attrs["none_value"] = "";
                attrs[PtFrameworkNode::failed_conversion_key] =
                    "None constant cannot be converted to OpenVINO opset and should be removed by consuming "
                    "operation.";
                fw_node->set_attrs(attrs);
                return fw_node->output(0);
            } else {
                auto inlined_decoder = m_decoder->get_inlined_input_decoder(index_);
                auto inlined_ctx = NodeContext(inlined_decoder,
                                               m_ext_tensor_map,
                                               m_tensor_map,
                                               m_external_parameters,
                                               m_mutated_tensors,
                                               m_translate_session);
                auto inlined_input = m_translate_session->convert_node(inlined_ctx);
                FRONT_END_GENERAL_CHECK(inlined_input.size() == 1,
                                        "Incorrect inlined input with index: ",
                                        index,
                                        " for operation ",
                                        get_op_type());
                return inlined_input[0];
            }
        }
    }
    auto tensor_it = m_tensor_map->find(input);
    FRONT_END_GENERAL_CHECK(tensor_it != m_tensor_map->end(), "No tensor corresponding input: ", input, " exist.");
    return resolve_tensor(input);
}

Output<Node> NodeContext::resolve_tensor(size_t index) const {
    const auto alias = m_translate_session->m_may_be_alias.find(index);
    if (alias != m_translate_session->m_may_be_alias.end() && alias->second.base_value.get_node()) {
        auto& info = alias->second;
        const auto base = resolve_tensor(info.base_id);
        if (base != info.base_value) {
            // Replaying a view against the current base updates sibling views and
            // views created before an in-place write. Previously consumed values
            // remain connected to their original OpenVINO nodes.
            info.output = rebase_view(info.output, info.base_value, base);
            info.base_value = base;
            (*m_tensor_map)[index] = info.output;
            m_translate_session->encode_tensor_name(info.output, index);
        }
    }
    return m_tensor_map->at(index);
}

Output<Node> NodeContext::get_input(const std::string& name) const {
    FRONT_END_GENERAL_CHECK(has_attribute(name), "Input with name ", name, " doesn't exist");
    auto attr = get_attribute_as_any(name);
    if (attr.is<Output<Node>>()) {
        // Case when input is constant value
        return attr.as<Output<Node>>();
    } else if (attr.is<type::PyNone>()) {
        // None means input is unknown type, most likely a Node
        auto input = m_decoder->get_named_input(name);
        FRONT_END_GENERAL_CHECK(m_tensor_map->count(input), "No tensor corresponding input: ", input, " exist.");
        return resolve_tensor(input);
    }
    FRONT_END_GENERAL_CHECK(false, "Input has type which can't be converted to ov::Node.");
}

OutputVector NodeContext::inputs() const {
    OutputVector res;
    for (size_t i = 0; i < m_decoder_inputs.size(); i++) {
        res.push_back(get_input(static_cast<int>(i)));
    }
    return res;
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
    auto input = ov::as_type_ptr<PtFrameworkNode>(input_node);
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
    if (auto input = ov::as_type_ptr<PtFrameworkNode>(input_val.get_node_shared_ptr())) {
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
        case element::f16:
            return get_constant_data<ov::float16>(constant);
        case element::bf16:
            return get_constant_data<ov::bfloat16>(constant);
        case element::boolean:
            return get_constant_data<bool>(constant);
        default:
            FRONT_END_GENERAL_CHECK(false, "Input with index: ", index, " has unsupported type.");
        }
    }
    FRONT_END_GENERAL_CHECK(false, "Input node with index ", index, " cannot be interpreted as constant", input_val);

    return 0;
}

ov::Any NodeContext::apply_additional_conversion_rules(const ov::Any& data, const std::type_info& type_info) const {
    if (data.is<Output<Node>>() && type_info != typeid(Output<Node>)) {
        auto const_node = as_type_ptr<v0::Constant>(data.as<Output<Node>>().get_node_shared_ptr());
        FRONT_END_GENERAL_CHECK(const_node, "Attribute must be const if requested as not a Node.");
        if (type_info == typeid(bool)) {
            bool res = const_node->cast_vector<bool>()[0];
            return res;
        } else if (type_info == typeid(int32_t)) {
            int32_t res = const_node->cast_vector<int32_t>()[0];
            return res;
        } else if (type_info == typeid(int64_t)) {
            int64_t res = const_node->cast_vector<int64_t>()[0];
            return res;
        } else if (type_info == typeid(double)) {
            double res = const_node->cast_vector<double>()[0];
            return res;
        } else if (type_info == typeid(float)) {
            float res = const_node->cast_vector<float>()[0];
            return res;
        } else {
            FRONT_END_GENERAL_CHECK(false,
                                    "Could not decode attribute for ",
                                    get_name(),
                                    " node. Provided type is not known.");
        }
    }
    return data;
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
