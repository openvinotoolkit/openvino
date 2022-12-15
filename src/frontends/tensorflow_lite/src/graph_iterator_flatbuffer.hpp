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

class GraphIteratorFlatBuffer {
    std::vector<const Operator*> m_nodes;
    size_t node_index = 0;
    std::shared_ptr<tflite::Model> m_model;
    std::vector<const tflite::Tensor *> m_tensors;
    std::vector<const tflite::Buffer *> m_buffers;

public:
    template <typename T>
    explicit GraphIteratorFlatBuffer(const std::basic_string<T>& path) {
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

        m_model = std::shared_ptr<tflite::Model>(GetMutableModel(data), [](tflite::Model* p){});
        const auto subgraphs = m_model->subgraphs();
        FRONT_END_GENERAL_CHECK(subgraphs->size() == 1,
                                "Number of sub-graphs in the model is ",
                                subgraphs->size(),
                                ". Supported number of sub-graphs is 1.");
        auto graph = *subgraphs->begin();

        const auto* operators = graph->operators();
        std::cout << "Num operations: " << operators->size() << std::endl;
        const auto tensors = graph->tensors();
        m_tensors = std::vector<const tflite::Tensor*>(tensors->begin(), tensors->end());

        std::cout << "Num tensors: " << tensors->size() << std::endl;
        std::cout << "Num tensors: " << tensors->size() << std::endl;
        const auto buffers = m_model->buffers();

        m_buffers = {buffers->begin(), buffers->end()};

        std::cout << "Num buffers: " << buffers->size() << std::endl;

        const auto& op_names = tflite::EnumNamesBuiltinOperator();
        auto opcodes = (m_model->operator_codes());

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
    using Ptr = std::shared_ptr<GraphIteratorFlatBuffer>;

    ~GraphIteratorFlatBuffer() = default;

    /// Set iterator to the start position
    void reset() {
        node_index = 0;
    }

    size_t size() const {
        return m_nodes.size();
    }

    /// Moves to the next node in the graph
    void next() {
        node_index++;
    }

    bool is_end() const {
        return node_index >= m_nodes.size();
    }

    std::vector<size_t> get_input_tensor_indices() const {
        auto inputs = (*(m_model->subgraphs()))[0]->inputs();
        return {inputs->begin(), inputs->end()};
    }

    std::vector<size_t> get_output_tensor_indices() const {
        auto outputs = (*(m_model->subgraphs()))[0]->outputs();
        return {outputs->begin(), outputs->end()};
    }

    const tflite::Tensor* get_tensor(size_t index) const {
        FRONT_END_GENERAL_CHECK(m_tensors.size() > index,
                                "Input tensor index is out of range. Tensor index: ",
                                index,
                                " Number of inputs: ",
                                m_tensors.size());
        return m_tensors[index];
    }

    std::vector<const tflite::Tensor *> get_tensors() const {
        return m_tensors;
    }

    std::vector<const tflite::Buffer *> get_buffers() const {
        return m_buffers;
    }

    /// Return NodeContext for the current node that iterator points to
    std::shared_ptr<DecoderFlatBuffer> get_decoder() const {
        std::string type = tflite::EnumNamesBuiltinOperator()[(*m_model->operator_codes())[m_nodes[node_index]->opcode_index()]->builtin_code()];
        return std::make_shared<DecoderFlatBuffer>(m_nodes[node_index], type, std::to_string(node_index), m_tensors);
    }
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
