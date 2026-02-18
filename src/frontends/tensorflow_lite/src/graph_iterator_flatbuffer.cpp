// Copyright (C) 2018-2026 Intel Corporation
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
    FRONT_END_GENERAL_CHECK(m_model != nullptr, "Failed to parse TFLite model from file: ", path);
    auto sub_graphs = m_model->subgraphs();
    FRONT_END_GENERAL_CHECK(sub_graphs && sub_graphs->size() > 0, "TFLite model has no subgraphs in file: ", path);
    m_subgraphs = {sub_graphs->begin(), sub_graphs->end()};
    m_graph = m_subgraphs[0];
    FRONT_END_GENERAL_CHECK(m_graph != nullptr, "First subgraph is null in file: ", path);
    const auto operators = m_graph->operators();
    FRONT_END_GENERAL_CHECK(operators != nullptr, "TFLite subgraph has no operators in file: ", path);
    auto operators_vec = std::vector<const tflite::Operator*>{operators->begin(), operators->end()};

    m_nodes.assign(operators_vec.begin(), operators_vec.end());
    auto outputs = m_graph->outputs();
    auto inputs = m_graph->inputs();
    FRONT_END_GENERAL_CHECK(outputs != nullptr, "TFLite subgraph has no outputs in file: ", path);
    FRONT_END_GENERAL_CHECK(inputs != nullptr, "TFLite subgraph has no inputs in file: ", path);
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
    FRONT_END_GENERAL_CHECK(iterator->m_graph != nullptr, "Subgraph at index ", idx, " is null");
    const auto operators = iterator->m_graph->operators();
    FRONT_END_GENERAL_CHECK(operators != nullptr, "TFLite subgraph has no operators");
    auto operators_vec = std::vector<const tflite::Operator*>{operators->begin(), operators->end()};
    iterator->m_nodes.assign(operators_vec.begin(), operators_vec.end());
    auto outputs = iterator->m_graph->outputs();
    auto inputs = iterator->m_graph->inputs();
    FRONT_END_GENERAL_CHECK(outputs != nullptr, "TFLite subgraph has no outputs");
    FRONT_END_GENERAL_CHECK(inputs != nullptr, "TFLite subgraph has no inputs");
    iterator->m_nodes.insert(iterator->m_nodes.begin(), outputs->begin(), outputs->end());
    iterator->m_nodes.insert(iterator->m_nodes.begin(), inputs->begin(), inputs->end());
    return iterator;
}

std::shared_ptr<DecoderBase> GraphIteratorFlatBuffer::get_decoder() const {
    auto any_item = m_nodes[node_index];
    bool is_op = any_item.is<const tflite::Operator*>();
    FRONT_END_GENERAL_CHECK(is_op || any_item.is<int32_t>());
    auto tensors = m_graph->tensors();
    FRONT_END_GENERAL_CHECK(tensors != nullptr, "TFLite subgraph has no tensors");
    const auto tensors_size = tensors->size();

    if (is_op) {
        auto node = m_nodes[node_index].as<const tflite::Operator*>();
        FRONT_END_GENERAL_CHECK(node != nullptr, "Null operator at node index ", node_index);
        auto buffers = m_model->buffers();
        FRONT_END_GENERAL_CHECK(buffers != nullptr, "TFLite model has no buffers");
        const auto buffers_size = buffers->size();

        std::map<size_t, TensorInfo> input_info = {}, output_info = {};
        size_t i = 0;
        FRONT_END_GENERAL_CHECK(node->inputs() != nullptr, "Operator has no inputs");
        for (auto input : *node->inputs()) {
            if (input == -1)
                continue;
            FRONT_END_GENERAL_CHECK(input >= 0 && static_cast<size_t>(input) < tensors_size,
                                    "Operator input tensor index ",
                                    input,
                                    " is out of range. Number of tensors: ",
                                    tensors_size);
            auto tensor = (*tensors)[input];
            FRONT_END_GENERAL_CHECK(tensor != nullptr, "Null tensor at index ", input);
            FRONT_END_GENERAL_CHECK(tensor->buffer() < buffers_size,
                                    "Tensor buffer index ",
                                    tensor->buffer(),
                                    " is out of range. Number of buffers: ",
                                    buffers_size);
            auto buffer = (*buffers)[tensor->buffer()];
            input_info[i++] = TensorInfo{tensor, buffer};
        }
        i = 0;
        FRONT_END_GENERAL_CHECK(node->outputs() != nullptr, "Operator has no outputs");
        for (auto output : *node->outputs()) {
            if (output == -1)
                continue;
            FRONT_END_GENERAL_CHECK(output >= 0 && static_cast<size_t>(output) < tensors_size,
                                    "Operator output tensor index ",
                                    output,
                                    " is out of range. Number of tensors: ",
                                    tensors_size);
            auto tensor = (*tensors)[output];
            FRONT_END_GENERAL_CHECK(tensor != nullptr, "Null tensor at index ", output);
            FRONT_END_GENERAL_CHECK(tensor->buffer() < buffers_size,
                                    "Tensor buffer index ",
                                    tensor->buffer(),
                                    " is out of range. Number of buffers: ",
                                    buffers_size);
            auto buffer = (*buffers)[tensor->buffer()];
            output_info[i++] = TensorInfo{tensor, buffer};
        }
        auto op_codes = m_model->operator_codes();
        FRONT_END_GENERAL_CHECK(op_codes != nullptr, "TFLite model has no operator codes");
        FRONT_END_GENERAL_CHECK(node->opcode_index() < op_codes->size(),
                                "Operator opcode index ",
                                node->opcode_index(),
                                " is out of range. Number of operator codes: ",
                                op_codes->size());
        auto operator_code = (*op_codes)[node->opcode_index()];
        FRONT_END_GENERAL_CHECK(operator_code != nullptr, "Null operator code at index ", node->opcode_index());
        std::string type;
        if (operator_code->deprecated_builtin_code() <
            tflite::BuiltinOperator::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
            type = tflite::EnumNamesBuiltinOperator()[operator_code->deprecated_builtin_code()];
        } else {
            type = tflite::EnumNamesBuiltinOperator()[operator_code->builtin_code()];
        }
        if (type == "CUSTOM") {
            auto custom_code = operator_code->custom_code();
            FRONT_END_GENERAL_CHECK(custom_code != nullptr, "Operator has CUSTOM type but no custom_code string");
            type = custom_code->str();
        }
        auto name = std::to_string(node_index - m_graph->inputs()->size() - m_graph->outputs()->size());
        return std::make_shared<DecoderFlatBuffer>(node, type, name, input_info, output_info);
    } else {
        auto tensor_id = m_nodes[node_index].as<int32_t>();
        FRONT_END_GENERAL_CHECK(tensor_id >= 0 && static_cast<size_t>(tensor_id) < tensors_size,
                                "Graph input/output tensor index ",
                                tensor_id,
                                " is out of range. Number of tensors: ",
                                tensors_size);
        auto tensor = (*tensors)[tensor_id];
        FRONT_END_GENERAL_CHECK(tensor != nullptr, "Null tensor at index ", tensor_id);
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
