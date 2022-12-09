// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "schema_generated.h"
//#include "decoder_proto.hpp"
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
    std::vector<tflite::Operator*> m_nodes;
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

        auto raw_model = GetMutableModel(data);
//        m_graph_def = std::shared_ptr<tflite::Model>(raw_model);

        std::cout << "Model of version " << raw_model->version() << " read from " << path << std::endl;
        std::cout << "Description: " << std::string(raw_model->description()->str()) << std::endl;

//        auto graph = m_graph_def->subgraphs()->begin();
//        std::cout << graph->name()->str() << std::endl;
//        const auto* inputs = graph->inputs();
//        const auto& input = *(inputs->begin());
//        const auto* outputs = graph->outputs();
//        const auto& output = *(outputs->begin());
//        const auto* operators = graph->operators();
//        const auto& operator_ = *(operators->begin());
//        const auto* tensors = graph->tensors();
//        const auto& tensor = *(tensors->begin());

        std::cout << "Success with: " << path << std::endl;
    }

    ~GraphIteratorFlatBuffer() {

    }

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
        //        return std::make_shared<DecoderProto>(m_nodes[node_index]);
        //      FIXME
        return nullptr;
    }
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
