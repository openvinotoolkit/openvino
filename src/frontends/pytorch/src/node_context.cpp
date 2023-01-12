// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/opsets/opset10.hpp"

#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

Output<Node> NodeContext::get_tensor_from_model_or_create_input(size_t index) {
    if (m_tensor_map->find(index) != m_tensor_map->end()) {
        return m_tensor_map->at(index);
    } else {
        // nested subgraphs case
        auto parameter = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        parameter->get_output_tensor(0).add_names({std::to_string(index)});
        (*m_tensor_map)[index] = parameter;
        m_external_parameters->push_back(parameter);
        // std::cout << "Nested case, created: " << parameter << std::endl;
        return parameter;
    }
}

Output<Node> NodeContext::get_input_from_visible_context(size_t index) const {
    FRONT_END_GENERAL_CHECK(index < get_input_size(), "Index is lower then number of inputs.");
    auto input_tensor = get_input(index);
    auto input_node = input_tensor.get_node_shared_ptr();
    if (std::dynamic_pointer_cast<opset10::Parameter>(input_node)) {
        // We need to look into external context for inputs that would be feed into this parameter
        auto name = input_node->get_output_tensor(0).get_any_name();
        size_t tensor_idx = (size_t)std::stoll(name);
        if (m_ext_tensor_map.count(tensor_idx)) {
            input_tensor = m_ext_tensor_map.at(tensor_idx);
        }
    }
    return input_tensor;
}

std::shared_ptr<ov::Model> NodeContext::convert_subgraph(size_t index) {
    auto subgraph_decoder = m_decoder->get_subgraph_decoder(index);

    // Extend external context with internal tensors except Parameter nodes, because internal Parameters are created to
    // link internal context with external
    TensorMap ext_map(m_ext_tensor_map);
    // map::insert does not update elements if their key is already in map; so if we have real tensors in outter scope
    // we will not add Parameters we creeated in inner scope.
    ext_map.insert(m_tensor_map->begin(), m_tensor_map->end());

    auto model = convert_pytorch_model(subgraph_decoder, ext_map);
    // Remove unused parameters, they could be created as inputs to the parts of graph that weren't
    // used for generating output.
    for (int i = subgraph_decoder->inputs().size(); i < model->get_parameters().size(); i++) {
        auto parameter = model->get_parameters()[i];
        if (parameter->output(0).get_target_inputs().empty()) {
            // There is no consumers: safe to remove
            // std::cout << "[ WARNING ] Removing parameter " << parameter
            //          << " in converted Pytorch model, because it is never used" << std::endl;
            model->remove_parameter(parameter);
        }
    }
    return model;
}

namespace {
std::shared_ptr<opset10::Constant> get_constant_at_input(const NodeContext& ctx, size_t index) {
    FRONT_END_GENERAL_CHECK(!ctx.input_is_none(index), "Input with index: ", index, " is none.");
    auto input_node = ctx.get_input_from_visible_context(index).get_node_shared_ptr();
    auto input = std::dynamic_pointer_cast<opset10::Constant>(input_node);
    FRONT_END_GENERAL_CHECK(input, "Input with index ", index, " cannot be interpreted as Constant: ", input_node);
    return input;
}
}  // namespace

template <>
std::vector<int64_t> NodeContext::const_input<std::vector<int64_t>>(size_t index) const {
    return get_constant_at_input(*this, index)->cast_vector<int64_t>();
}

template <>
ngraph::Strides NodeContext::const_input<ngraph::Strides>(size_t index) const {
    return get_constant_at_input(*this, index)->cast_vector<ngraph::Strides::value_type>();
}

template <>
ngraph::CoordinateDiff NodeContext::const_input<ngraph::CoordinateDiff>(size_t index) const {
    return get_constant_at_input(*this, index)->cast_vector<ngraph::CoordinateDiff::value_type>();
}

template <>
ngraph::Shape NodeContext::const_input<ngraph::Shape>(size_t index) const {
    return get_constant_at_input(*this, index)->cast_vector<ngraph::Shape::value_type>();
}

template <>
int64_t NodeContext::const_input<int64_t>(size_t index) const {
    return get_constant_at_input(*this, index)->cast_vector<int64_t>()[0];
}

template <>
bool NodeContext::const_input<bool>(size_t index) const {
    return get_constant_at_input(*this, index)->cast_vector<bool>()[0];
}

template <>
double NodeContext::const_input<double>(size_t index) const {
    return get_constant_at_input(*this, index)->cast_vector<double>()[0];
}

template <>
float NodeContext::const_input<float>(size_t index) const {
    return get_constant_at_input(*this, index)->cast_vector<float>()[0];
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

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
