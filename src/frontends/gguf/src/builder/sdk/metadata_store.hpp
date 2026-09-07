// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Internal storage behind the opaque SDK metadata and weight views.

#pragma once

#include <map>
#include <string>
#include <unordered_map>

#include "openvino/frontend/gguf/builder/metadata.hpp"
#include "openvino/runtime/tensor.hpp"
#include "quant/gguf.hpp"

namespace ov {
namespace frontend {
namespace gguf {

class GraphEmitter;

namespace detail {

// The parsed KV metadata, behind GgufMetadata.
struct MetadataStore {
    const std::unordered_map<std::string, GGUFMetaData>& map;
};

struct MetadataAccess {
    static const MetadataStore& get(const GgufMetadata& metadata) {
        return *metadata.m_store;
    }
};

// Borrowed parser weight and quantization tables.
struct WeightStore {
    std::unordered_map<std::string, ov::Tensor>& weights;
    std::unordered_map<std::string, GgufTensorType>& qtypes;
};

}  // namespace detail
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
