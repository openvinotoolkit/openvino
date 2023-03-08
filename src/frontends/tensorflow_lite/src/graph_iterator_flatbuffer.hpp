// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "decoder_flatbuffer.h"
#include "openvino/frontend/exception.hpp"
#include "openvino/util/file_util.hpp"
#include "schema_generated.h"

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
    std::vector<const tflite::Operator*> m_nodes;
    std::shared_ptr<tflite::Model> m_model;

public:
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
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
