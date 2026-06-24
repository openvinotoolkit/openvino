// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_flatbuffer.hpp"

#include <map>

#include "decoder_flatbuffer.h"

using namespace ov::frontend::tensorflow_lite;

namespace {
enum class TensorKind { INPUT, OUTPUT };

std::map<size_t, TensorInfo> collect_tensor_info(
    const flatbuffers::Vector<int32_t>* indices,
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>* tensors,
    size_t tensors_size,
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers,
    size_t buffers_size,
    TensorKind kind) {
    const auto kind_str = kind == TensorKind::INPUT ? "input" : "output";
    FRONT_END_GENERAL_CHECK(indices != nullptr, "Operator has no ", kind_str, "s");
    std::map<size_t, TensorInfo> info;
    size_t i = 0;
    for (auto idx : *indices) {
        if (idx == -1)
            continue;
        FRONT_END_GENERAL_CHECK(idx >= 0 && static_cast<size_t>(idx) < tensors_size,
                                "Operator ",
                                kind_str,
                                " tensor index ",
                                idx,
                                " is out of range. Number of tensors: ",
                                tensors_size);
        auto tensor = (*tensors)[idx];
        FRONT_END_GENERAL_CHECK(tensor != nullptr, "Null tensor at index ", idx);
        FRONT_END_GENERAL_CHECK(tensor->buffer() < buffers_size,
                                "Tensor buffer index ",
                                tensor->buffer(),
                                " is out of range. Number of buffers: ",
                                buffers_size);
        info[i++] = TensorInfo{tensor, (*buffers)[tensor->buffer()]};
    }
    return info;
}

void populate_nodes(const tflite::SubGraph* graph, std::vector<ov::Any>& nodes) {
    const auto operators = graph->operators();
    FRONT_END_GENERAL_CHECK(operators != nullptr, "TFLite subgraph has no operators");
    nodes.assign(operators->begin(), operators->end());
    const auto outputs = graph->outputs();
    const auto inputs = graph->inputs();
    FRONT_END_GENERAL_CHECK(outputs != nullptr, "TFLite subgraph has no outputs");
    FRONT_END_GENERAL_CHECK(inputs != nullptr, "TFLite subgraph has no inputs");
    nodes.insert(nodes.begin(), outputs->begin(), outputs->end());
    nodes.insert(nodes.begin(), inputs->begin(), inputs->end());
}

std::string get_builtin_operator_type(int32_t builtin_code) {
    FRONT_END_GENERAL_CHECK(builtin_code >= tflite::BuiltinOperator_MIN && builtin_code <= tflite::BuiltinOperator_MAX,
                            "Operator builtin code ",
                            builtin_code,
                            " is out of range [",
                            static_cast<int>(tflite::BuiltinOperator_MIN),
                            ", ",
                            static_cast<int>(tflite::BuiltinOperator_MAX),
                            "].");
    return tflite::EnumNamesBuiltinOperator()[builtin_code];
}
}  // namespace

GraphIteratorFlatBuffer::GraphIteratorFlatBuffer(const std::filesystem::path& path) {
    std::ifstream model_file(path, std::ios::binary | std::ios::in);
    FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Model file does not exist: ", path);

    m_data = {(std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>()};
    model_file.close();

    flatbuffers::Verifier verifier(m_data.data(), m_data.size());
    FRONT_END_GENERAL_CHECK(tflite::VerifyModelBuffer(verifier),
                            "TensorFlow Lite Frontend: the model file ",
                            path,
                            " is corrupted or malformed (FlatBuffer verification failed).");

    m_model = tflite::GetModel(m_data.data());
    FRONT_END_GENERAL_CHECK(m_model != nullptr, "Failed to parse TFLite model from file: ", path);
    auto sub_graphs = m_model->subgraphs();
    FRONT_END_GENERAL_CHECK(sub_graphs && sub_graphs->size() > 0, "TFLite model has no subgraphs in file: ", path);
    m_subgraphs = {sub_graphs->begin(), sub_graphs->end()};
    m_graph = m_subgraphs[0];
    FRONT_END_GENERAL_CHECK(m_graph != nullptr, "First subgraph is null in file: ", path);
    populate_nodes(m_graph, m_nodes);
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
    populate_nodes(iterator->m_graph, iterator->m_nodes);
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

        auto input_info =
            collect_tensor_info(node->inputs(), tensors, tensors_size, buffers, buffers_size, TensorKind::INPUT);
        auto output_info =
            collect_tensor_info(node->outputs(), tensors, tensors_size, buffers, buffers_size, TensorKind::OUTPUT);
        auto op_codes = m_model->operator_codes();
        FRONT_END_GENERAL_CHECK(op_codes != nullptr, "TFLite model has no operator codes");
        FRONT_END_GENERAL_CHECK(node->opcode_index() < op_codes->size(),
                                "Operator opcode index ",
                                node->opcode_index(),
                                " is out of range. Number of operator codes: ",
                                op_codes->size());
        auto operator_code = (*op_codes)[node->opcode_index()];
        FRONT_END_GENERAL_CHECK(operator_code != nullptr, "Null operator code at index ", node->opcode_index());
        const auto deprecated_code = operator_code->deprecated_builtin_code();
        std::string type = deprecated_code < tflite::BuiltinOperator::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
                               ? get_builtin_operator_type(deprecated_code)
                               : get_builtin_operator_type(operator_code->builtin_code());
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
