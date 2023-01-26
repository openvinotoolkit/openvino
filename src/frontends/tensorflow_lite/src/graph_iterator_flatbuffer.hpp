// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "decoder_flatbuffer.h"
#include "openvino/frontend/exception.hpp"
#include "openvino/util/file_util.hpp"
#include "schema_generated.h"

using namespace tflite;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
class DecoderFlatBuffer;

struct TensorInfo {
    int64_t input_idx, output_idx;
    const tflite::Tensor* tensor;
    const tflite::Buffer* buffer;
};

class GraphIteratorFlatBuffer {
    size_t node_index = 0;
    std::vector<const Operator*> m_nodes;
    std::shared_ptr<tflite::Model> m_model;

public:
    template <typename T>
    explicit GraphIteratorFlatBuffer(const std::basic_string<T>& path) {
        std::ifstream model_file;
        model_file.open(path, std::ios::binary | std::ios::in);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Model file does not exist: ", ov::util::wstring_to_string(path));
#else
        FRONT_END_GENERAL_CHECK(model_file && model_file.is_open(), "Model file does not exist: ", path);
#endif

        model_file.seekg(0, std::ios::end);
        auto length = model_file.tellg();
        model_file.seekg(0, std::ios::beg);
        char* data = new char[length];
        model_file.read(data, length);
        model_file.close();

        m_model = std::shared_ptr<tflite::Model>(GetMutableModel(data), [](tflite::Model* p) {});
        const auto subgraphs = m_model->subgraphs();
        FRONT_END_GENERAL_CHECK(subgraphs->size() == 1,
                                "Number of sub-graphs in the model is ",
                                subgraphs->size(),
                                ". Supported number of sub-graphs is 1.");
        const auto graph = *subgraphs->begin();
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

    /// Return Decoder for the current node that iterator points to
    std::shared_ptr<ov::frontend::tensorflow_lite::DecoderFlatBuffer> get_decoder() const;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
