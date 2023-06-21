// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "openvino/core/any.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/util/file_util.hpp"
#include "schema_generated.h"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
class DecoderFlatBuffer;

struct TensorInfo {
    const tflite::Tensor* tensor;
    const tflite::Buffer* buffer;
};

class GraphIteratorFlatBuffer {
    size_t node_index = 0;
    std::vector<uint8_t> m_data;
    std::vector<ov::Any> m_nodes;
    const tflite::Model* m_model{};
    std::vector<const tflite::SubGraph*> m_subgraphs;
    const tflite::SubGraph* m_graph{};

public:
    GraphIteratorFlatBuffer() = default;
    explicit GraphIteratorFlatBuffer(const std::string& path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    explicit GraphIteratorFlatBuffer(const std::wstring& path);
#endif

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

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    size_t get_subgraph_size() const;

    /// \brief Returns iterator for a subgraph created on demand
    /// If there is no query for specific sub-graph iterator shouldn't be created
    /// idx should be in range 0..get_subgraph_size()-1
    std::shared_ptr<GraphIteratorFlatBuffer> get_subgraph(const size_t& idx) const;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
