// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_flatbuffer.hpp"

#include <map>

#include "decoder_flatbuffer.h"

using namespace ov::frontend::tensorflow_lite;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

GraphIteratorFlatBuffer::GraphIteratorFlatBuffer(const std::wstring& path)
    : GraphIteratorFlatBuffer(ov::util::wstring_to_string(path)) {}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

GraphIteratorFlatBuffer::GraphIteratorFlatBuffer(const std::string& path) {
    std::ifstream model_file(path, std::ios::binary | std::ios::in);
    FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Model file does not exist: ", path);

    m_data = {(std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>()};
    model_file.close();

    m_model = tflite::GetModel(m_data.data());
    auto sub_graphs = m_model->subgraphs();
    m_subgraphs = {sub_graphs->begin(), sub_graphs->end()};
    m_graph = m_subgraphs[0];
    const auto operators = m_graph->operators();
    auto operators_vec = std::vector<const tflite::Operator*>{operators->begin(), operators->end()};

    m_nodes.assign(operators_vec.begin(), operators_vec.end());
    auto outputs = m_graph->outputs();
    auto inputs = m_graph->inputs();
    m_nodes.insert(m_nodes.begin(), outputs->begin(), outputs->end());
    m_nodes.insert(m_nodes.begin(), inputs->begin(), inputs->end());
}

size_t GraphIteratorFlatBuffer::get_subgraph_size() const {
    return m_subgraphs.size();
}

std::shared_ptr<GraphIterator> GraphIteratorFlatBuffer::get_subgraph(size_t idx) const {
    FRONT_END_GENERAL_CHECK(m_subgraphs.size() > idx, "There is no subgraph with idx ", idx);
    auto iterator = std::make_shared<GraphIteratorFlatBuffer>();
    iterator->node_index = 0;
    iterator->m_model = m_model;
    iterator->m_subgraphs = {};  // TODO: check if we need to pass all sub-graphs here (while in a while situation)
    iterator->m_graph = m_subgraphs[idx];
    const auto operators = iterator->m_graph->operators();
    auto operators_vec = std::vector<const tflite::Operator*>{operators->begin(), operators->end()};
    iterator->m_nodes.assign(operators_vec.begin(), operators_vec.end());
    auto outputs = iterator->m_graph->outputs();
    auto inputs = iterator->m_graph->inputs();
    iterator->m_nodes.insert(iterator->m_nodes.begin(), outputs->begin(), outputs->end());
    iterator->m_nodes.insert(iterator->m_nodes.begin(), inputs->begin(), inputs->end());
    return iterator;
}

std::shared_ptr<DecoderBase> GraphIteratorFlatBuffer::get_decoder() const {
    auto any_item = m_nodes[node_index];
    bool is_op = any_item.is<const tflite::Operator*>();
    FRONT_END_GENERAL_CHECK(is_op || any_item.is<int32_t>());
    auto tensors = m_graph->tensors();

    if (is_op) {
        auto node = m_nodes[node_index].as<const tflite::Operator*>();
        auto buffers = m_model->buffers();

        std::map<size_t, TensorInfo> input_info = {}, output_info = {};
        size_t i = 0;
        for (auto input : *node->inputs()) {
            if (input == -1)
                continue;
            auto buffer = (*buffers)[(*tensors)[input]->buffer()];
            auto tensor = (*tensors)[input];
            input_info[i++] = TensorInfo{tensor, buffer};
        }
        i = 0;
        for (auto output : *node->outputs()) {
            auto buffer = (*buffers)[(*tensors)[output]->buffer()];
            auto tensor = (*tensors)[output];
            output_info[i++] = TensorInfo{tensor, buffer};
        }
        auto op_codes = m_model->operator_codes();
        auto operator_code = (*op_codes)[node->opcode_index()];
        std::string type;
        if (operator_code->deprecated_builtin_code() <
            tflite::BuiltinOperator::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
            type = tflite::EnumNamesBuiltinOperator()[operator_code->deprecated_builtin_code()];
        } else {
            type = tflite::EnumNamesBuiltinOperator()[operator_code->builtin_code()];
        }
        if (type == "CUSTOM") {
            type = operator_code->custom_code()->str();
        }
        auto name = std::to_string(node_index - m_graph->inputs()->size() - m_graph->outputs()->size());
        return std::make_shared<DecoderFlatBuffer>(node, type, name, input_info, output_info);
    } else {
        auto tensor_id = m_nodes[node_index].as<int32_t>();
        auto tensor = (*tensors)[tensor_id];
        auto info = TensorInfo{tensor, nullptr};
        auto inputs = m_graph->inputs();
        auto outputs = m_graph->outputs();

        auto input_it = std::find(inputs->begin(), inputs->end(), tensor_id);
        auto output_it = std::find(outputs->begin(), outputs->end(), tensor_id);
        int64_t input_idx =
            input_it == inputs->end() ? -1 : static_cast<int64_t>(std::distance(inputs->begin(), input_it));
        int64_t output_idx =
            output_it == outputs->end() ? -1 : static_cast<int64_t>(std::distance(outputs->begin(), output_it));
        return std::make_shared<DecoderFlatBufferTensors>(info, input_idx, output_idx);
    }
}

template <>
std::basic_string<char> ov::frontend::tensorflow_lite::get_model_extension<char>() {
    return ::tflite::ModelExtension();
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> ov::frontend::tensorflow_lite::get_model_extension<wchar_t>() {
    return util::string_to_wstring(::tflite::ModelExtension());
}
#endif
