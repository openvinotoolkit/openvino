// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <fstream>

#include "openvino/core/any.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "openvino/util/file_util.hpp"
#include "schema_generated.h"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

struct TensorInfo {
    const tflite::Tensor* tensor;
    const tflite::Buffer* buffer;
};

class GraphIteratorFlatBuffer : public GraphIterator {
    size_t node_index = 0;
    std::vector<uint8_t> m_data;
    std::vector<ov::Any> m_nodes;
    const tflite::Model* m_model{};
    std::vector<const tflite::SubGraph*> m_subgraphs;
    const tflite::SubGraph* m_graph{};

public:
    GraphIteratorFlatBuffer() = default;
    explicit GraphIteratorFlatBuffer(const std::filesystem::path& path);

    using Ptr = std::shared_ptr<GraphIteratorFlatBuffer>;

    ~GraphIteratorFlatBuffer() = default;

    /// Verifies file is supported
    static bool is_supported(const std::filesystem::path& path) {
        FRONT_END_GENERAL_CHECK(util::file_exists(path), "Could not open the file: ", path);

        try {
            if (path.extension() != std::filesystem::path("." + std::string(::tflite::ModelExtension()))) {
                return false;
            }
            const std::streamsize offset_size = static_cast<std::streamsize>(sizeof(::flatbuffers::uoffset_t));
            std::streamsize file_size = util::file_size(path);
            // Skip files which less than size of file identifier
            if (file_size < offset_size) {
                return false;
            }
            std::ifstream tflite_stream(path, std::ios::in | std::ifstream::binary);
            char buf[offset_size * 2 + 1] = {};  // +1 is used to overcome gcc's -Wstringop-overread warning
            tflite_stream.read(buf, offset_size * 2);
            // If we have enough read bytes - try to detect prefixed identifier, else try without size prefix
            if ((tflite_stream.gcount() == offset_size * 2) && ::tflite::ModelBufferHasIdentifier(buf + offset_size)) {
                return true;
            } else if (tflite_stream.gcount() >= offset_size && ::tflite::ModelBufferHasIdentifier(buf)) {
                return true;
            }
            return false;
        } catch (...) {
            return false;
        }
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

    /// Return Decoder for the current node that iterator points to
    std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase> get_decoder() const override;

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    size_t get_subgraph_size() const override;

    /// \brief Returns iterator for a subgraph created on demand
    /// If there is no query for specific sub-graph iterator shouldn't be created
    /// idx should be in range 0..get_subgraph_size()-1
    std::shared_ptr<GraphIterator> get_subgraph(size_t idx) const override;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
