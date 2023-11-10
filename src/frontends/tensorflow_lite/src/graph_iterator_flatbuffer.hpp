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

    /// Verifies file is supported
    template <typename T>
    static bool is_supported(const std::basic_string<T>& path) {
        try {
            if (!ov::util::ends_with(path, get_model_extension<T>())) {
                return false;
            }
            const size_t offset_size = sizeof(::flatbuffers::uoffset_t);
            size_t file_size = util::file_size(path);
            // Skip files which less than size of file identifier
            if (file_size < offset_size) {
                return false;
            }
            std::ifstream tflite_stream(path, std::ios::in | std::ifstream::binary);
            char buf[offset_size * 2] = {};
            tflite_stream.read(buf, offset_size * 2);
            // If we have enough readed bytes - try to detect prefixed identifier, else try without size prefix
            if ((tflite_stream.gcount() == offset_size * 2) && ::tflite::SizePrefixedModelBufferHasIdentifier(buf)) {
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

private:
    template <typename T>
    static std::basic_string<T> get_model_extension();

    template <>
    static std::basic_string<char> get_model_extension<char>() {
        return ::tflite::ModelExtension();
    }

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    template <>
    static std::basic_string<wchar_t> get_model_extension<wchar_t>() {
        return util::string_to_wstring(::tflite::ModelExtension());
    }
#endif
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
