// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "schema_generated.h"
#include "decoder_flatbuffer.h"
#include "flatbuffers/flatbuffers.h"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"

using namespace tflite;

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class GraphIteratorFlatBuffer {
    size_t node_index = 0;
    std::vector<const Operator*> m_nodes;
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
        auto length = model_file.tellg();
        model_file.seekg(0, std::ios::beg);
        char* data = new char[length];
        model_file.read(data, length);
        model_file.close();

        m_model = std::shared_ptr<tflite::Model>(GetMutableModel(data), [](tflite::Model* p){});
        const auto subgraphs = m_model->subgraphs();
        FRONT_END_GENERAL_CHECK(subgraphs->size() == 1,
                                "Number of sub-graphs in the model is ",
                                subgraphs->size(),
                                ". Supported number of sub-graphs is 1.");
        const auto graph = *subgraphs->begin();
        const auto tensors = graph->tensors();
        m_tensors = {tensors->begin(), tensors->end()};
        const auto buffers = m_model->buffers();
        m_buffers = {buffers->begin(), buffers->end()};
        const auto operators = graph->operators();
        m_nodes = {operators->begin(), operators->end()};
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

    std::vector<size_t> get_model_input_tensor_indices() const {
        auto inputs = (*m_model->subgraphs()->begin())->inputs();
        return {inputs->begin(), inputs->end()};
    }

    std::vector<size_t> get_model_output_tensor_indices() const {
        auto outputs = (*m_model->subgraphs()->begin())->outputs();
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
        auto op_codes = m_model->operator_codes();
        auto operator_code = (*op_codes)[m_nodes[node_index]->opcode_index()];
        std::string type;
        if (operator_code->deprecated_builtin_code() < tflite::BuiltinOperator::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
            type = tflite::EnumNamesBuiltinOperator()[operator_code->deprecated_builtin_code()];
        } else {
            type = tflite::EnumNamesBuiltinOperator()[operator_code->builtin_code()];
        }
        return std::make_shared<DecoderFlatBuffer>(m_nodes[node_index], type, std::to_string(node_index), m_tensors);
    }
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
