// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <onnx/onnx_pb.h>

#include <openvino/frontend/graph_iterator.hpp>

#include "openvino/frontend/onnx/decoder.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/util/wstring_convert_util.hpp"

using ::ONNX_NAMESPACE::AttributeProto_AttributeType;
using ::ONNX_NAMESPACE::GraphProto;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::NodeProto;
using ::ONNX_NAMESPACE::OperatorSetIdProto;
using ::ONNX_NAMESPACE::TensorProto;
using ::ONNX_NAMESPACE::TensorProto_DataLocation;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ::ONNX_NAMESPACE::ValueInfoProto;
using ::ONNX_NAMESPACE::Version;

namespace ov {
namespace frontend {
namespace onnx {

class DecoderProtoTensor;

class GraphIteratorProto : public ov::frontend::onnx::GraphIterator {
    size_t node_index = 0;
    std::shared_ptr<ModelProto> m_model;
    const GraphProto* m_graph{};
    GraphIteratorProto* m_parent;
    std::vector<std::shared_ptr<ov::frontend::onnx::DecoderBase>> m_decoders{};
    std::map<std::string, std::shared_ptr<DecoderProtoTensor>> m_tensors{};

public:
    GraphIteratorProto() = default;
    explicit GraphIteratorProto(const std::string& path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    explicit GraphIteratorProto(const std::wstring& path);
#endif

    explicit GraphIteratorProto(GraphIteratorProto* parent, const GraphProto* graph_def);

    using Ptr = std::shared_ptr<GraphIteratorProto>;

    ~GraphIteratorProto() = default;

    /// Verifies file is supported
    template <typename T>
    static bool is_supported(const std::basic_string<T>& path) {
        FRONT_END_GENERAL_CHECK(util::file_exists(path),
                                "Could not open the file: \"",
                                util::path_to_string(path),
                                '"');
        try {
            std::streamsize file_size = util::file_size(path);
            // Skip files which less than size of file identifier
            if (file_size < 1) {
                return false;
            }
#if defined(__MINGW32__) || defined(__MINGW64__)
            std::ifstream tflite_stream(std::filesystem::path(path), std::ios::in | std::ifstream::binary);
#else
            std::ifstream tflite_stream(path, std::ios::in | std::ifstream::binary);
#endif
            // the model usually starts with a 0x08 byte indicating the ir_version value
            // so this checker expects at least 3 valid ONNX keys to be found in the validated model
            const size_t EXPECTED_FIELDS_FOUND = 3u;
            std::unordered_set<::onnx::Field, std::hash<int>> onnx_fields_found = {};
            try {
                while (!model.eof() && onnx_fields_found.size() < EXPECTED_FIELDS_FOUND) {
                    const auto field = ::onnx::decode_next_field(model);

                    if (onnx_fields_found.count(field.first) > 0) {
                        // if the same field is found twice, this is not a valid ONNX model
                        return false;
                    } else {
                        onnx_fields_found.insert(field.first);
                        ::onnx::skip_payload(model, field.second);
                    }
                }

                return onnx_fields_found.size() == EXPECTED_FIELDS_FOUND;
            } catch (...) {
                return false;
            }
        } catch (...) {
            return false;
        }
    }

    /// Set iterator to the start position
    void reset() override;

    size_t size() const override {
        return m_decoders.size();
    }

    /// Moves to the next node in the graph
    void next() override {
        node_index++;
    }

    bool is_end() const override {
        return node_index >= m_decoders.size();
    }

    /// Return Decoder for the current node that iterator points to
    std::shared_ptr<ov::frontend::onnx::DecoderBase> get_decoder() const override;

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    size_t get_subgraph_size() const override;

    /// \brief Returns iterator for a subgraph created on demand
    /// If there is no query for specific sub-graph iterator shouldn't be created
    /// idx should be in range 0..get_subgraph_size()-1
    std::shared_ptr<ov::frontend::onnx::GraphIterator> get_subgraph(size_t idx) const override;

    const GraphProto* get_graph() const {
        return m_graph;
    }

    std::int64_t get_opset_version(const std::string& domain) const override;

protected:
    /// \brief Returns DecoderProtoTensor found in the current scope, or in a parent scope
    /// \param name Name of tensor
    /// \param owner Returns real owner of the tensor
    std::shared_ptr<DecoderProtoTensor> get_tensor(const std::string& name, GraphIteratorProto** owner);
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
