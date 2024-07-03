// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#if defined(__MINGW32__) || defined(__MINGW64__)
#    include <filesystem>
#endif

#include "openvino/core/any.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "schema_generated.h"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

struct TensorInfo {
    const tflite::Tensor* tensor;
    const tflite::Buffer* buffer;
};

template <typename T>
std::basic_string<T> get_model_extension() {}
template <>
std::basic_string<char> get_model_extension<char>();

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_model_extension<wchar_t>();
#endif

class GraphIteratorFlatBuffer : public GraphIterator {
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
        FRONT_END_GENERAL_CHECK(util::file_exists(path),
                                "Could not open the file: \"",
                                util::path_to_string(path),
                                '"');
        try {
            if (!ov::util::ends_with<T>(path, get_model_extension<T>())) {
                return false;
            }
            const std::streamsize offset_size = static_cast<std::streamsize>(sizeof(::flatbuffers::uoffset_t));
            std::streamsize file_size = util::file_size(path);
            // Skip files which less than size of file identifier
            if (file_size < offset_size) {
                return false;
            }
#if defined(__MINGW32__) || defined(__MINGW64__)
            std::ifstream tflite_stream(std::filesystem::path(path), std::ios::in | std::ifstream::binary);
#else
            std::ifstream tflite_stream(path, std::ios::in | std::ifstream::binary);
#endif
            char buf[offset_size * 2 + 1] = {};  // +1 is used to overcome gcc's -Wstringop-overread warning
            tflite_stream.read(buf, offset_size * 2);
            // If we have enough readed bytes - try to detect prefixed identifier, else try without size prefix
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
