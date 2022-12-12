// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "schema_generated.h"
#include "decoder_flatbuffer.h"
//#include "graph.pb.h"
//#include "node_def.pb.h"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"

using namespace tflite;

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class GraphIteratorFlatBuffer : public tensorflow::GraphIterator {
    std::vector<const Operator*> m_nodes;
    size_t node_index = 0;
    std::shared_ptr<tflite::Model> m_graph_def;

public:
    template <typename T>
    GraphIteratorFlatBuffer(const std::basic_string<T>& path) {
        std::ifstream model_file;
        model_file.open(path, std::ios::binary | std::ios::in);
        FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Model file does not exist: ", path);

        model_file.seekg(0, std::ios::end);
        int length = model_file.tellg();
        model_file.seekg(0, std::ios::beg);
        char* data = new char[length];
        model_file.read(data, length);
        model_file.close();

        m_graph_def = std::shared_ptr<tflite::Model>(GetMutableModel(data), [](tflite::Model* p){});
        const auto subgraphs = m_graph_def->subgraphs();
        FRONT_END_GENERAL_CHECK(subgraphs->size() == 1,
                                "Number of sub-graphs in the model is ",
                                subgraphs->size(),
                                ". Supported number of sub-graphs is 1.");
        auto graph = *subgraphs->begin();
        const auto* operators = graph->operators();
        m_nodes.reserve(operators->size());
        std::cout << graph->name()->str() << std::endl;

        auto opcodes = (m_graph_def->operator_codes());
        for (const auto& code : *opcodes) {
            std::cout << "build_in code: " << code->builtin_code() << std::endl;
            std::cout << "version: " << code->version() << std::endl;
        }
        for (const auto* sign : *m_graph_def->signature_defs()) {
            std::cout << "sign " << sign->signature_key() << std::endl;
        }

        const auto& op_names = tflite::EnumNamesBuiltinOperator();
        for (const auto& node : *operators) {
            m_nodes.push_back(node);
            std::cout << node->opcode_index() << std::endl;
            std::cout << (*opcodes)[node->opcode_index()]->version() << std::endl;
            std::cout << (*opcodes)[node->opcode_index()]->builtin_code() << std::endl;
            std::cout << op_names[(*opcodes)[node->opcode_index()]->builtin_code()] << std::endl;
        }

//        const auto* inputs = graph->inputs();
//        const auto& input = *(inputs->begin());
//        const auto* outputs = graph->outputs();
//        const auto& output = *(outputs->begin());
//        const auto* operators = graph->operators();
//        const auto& operator_ = *(operators->begin());
//        const auto* tensors = graph->tensors();
//        const auto& tensor = *(tensors->begin());

    }

    ~GraphIteratorFlatBuffer() = default;

    /// Set iterator to the start position
    void reset() override {
        node_index = 0;
    }

    size_t size() const override {
        return m_nodes.size();
    }

    /// Moves to the next node in the graph
    void next() override {
        node_index++;
    }

    bool is_end() const override {
        return node_index >= m_nodes.size();
    }

    /// Return NodeContext for the current node that iterator points to
    std::shared_ptr<tensorflow::DecoderBase> get_decoder() const override {
        std::string type = tflite::EnumNamesBuiltinOperator()[(*m_graph_def->operator_codes())[m_nodes[node_index]->opcode_index()]->builtin_code()];
        return std::make_shared<DecoderFlatBuffer>(m_nodes[node_index], type, std::to_string(node_index));
    }
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
