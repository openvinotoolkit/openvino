// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "schema_generated.h"
#include "flatbuffers/flatbuffers.h"

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

void print_buffer(const tflite::TensorType& tf_type, const ov::Shape& ov_shape, const void* my_buffer) {
    if (tf_type == tflite::TensorType_INT32) {
        auto vec_ = ov::op::v0::Constant::create(element::i32, ov_shape, my_buffer)->cast_vector<int32_t>();
        size_t num = 0;
        for (const auto &elem: vec_) {
            if (num++ > 24)
                break;
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    } else if (tf_type == tflite::TensorType_UINT8) {
        auto vec_ = ov::op::v0::Constant::create(element::u8, ov_shape, my_buffer)->cast_vector<int32_t>();
        size_t num = 0;
        for (const auto &elem: vec_) {
            if (num++ > 24)
                break;
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Buffer of unknown type: " << EnumNameTensorType(tf_type) << std::endl;
    }
}
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
        int i = 0;

        m_graph_def = std::shared_ptr<tflite::Model>(GetMutableModel(data), [](tflite::Model* p){});
        const auto subgraphs = m_graph_def->subgraphs();
        FRONT_END_GENERAL_CHECK(subgraphs->size() == 1,
                                "Number of sub-graphs in the model is ",
                                subgraphs->size(),
                                ". Supported number of sub-graphs is 1.");
        auto graph = *subgraphs->begin();

        const auto* operators = graph->operators();
        std::cout << "Num operations: " << operators->size() << std::endl;
        const auto tensors = graph->tensors();
        std::cout << "Num tensors: " << tensors->size() << std::endl;
        std::cout << "Num tensors: " << tensors->size() << std::endl;
        const auto buffers = m_graph_def->buffers();
        std::cout << "Num buffers: " << buffers->size() << std::endl;

        const auto& op_names = tflite::EnumNamesBuiltinOperator();
        auto opcodes = (m_graph_def->operator_codes());

        const auto inputs = graph->inputs();
        const auto outputs = graph->outputs();
        std::unordered_set<int> intermediates;
        i = 0;
        for (const auto& node : *operators) {
            std::cout << "Op #" << i++ << ": " << op_names[(*opcodes)[node->opcode_index()]->builtin_code()] << std::endl;
            m_nodes.push_back(node);
            std::cout << "    Inputs: " << std::endl;
            for (const auto& input : *node->inputs()) {
                bool model_input = std::find(inputs->begin(), inputs->end(), input) != inputs->end();
                auto tensor = (*tensors)[input];
                const auto& tf_shape = tensor->shape();
                const auto& ov_shape = ov::Shape{tf_shape->begin(), tf_shape->end()};
                const auto& tf_type = tensor->type();
                const auto& name = tensor->name()->str();
                std::cout << "        Tensor #" << input << ": " << "shape " << ov_shape << " type: " << EnumNameTensorType(tf_type) << " name: " << name << " " << (model_input ? " model input" : "") << std::endl;

                auto my_buffer = (*buffers)[tensor->buffer()]->data()->data();
                bool buffer_has_no_data = input == 0;
                if (buffer_has_no_data || intermediates.find(input) != intermediates.end()) {
                    std::cout << "        No data in the buffer!" << std::endl;
                } else {
                    std::cout << "        Buffer #" << tensor->buffer() << std::endl;
                    print_buffer(tf_type, ov_shape, my_buffer);
                }
            }
            std::cout << "Outputs: " << std::endl;
            for (const auto& output : *node->outputs()) {
                std::cout << output << std::endl;
                intermediates.insert(output);
            }
        }

        m_nodes.reserve(operators->size());
        for (const auto& node : *operators) {
            m_nodes.push_back(node);
        }
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
