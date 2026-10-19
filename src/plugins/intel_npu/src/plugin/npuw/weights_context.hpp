// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "openvino/runtime/file_handle.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
class Node;

namespace npuw {
namespace s11n {

class PairHash {
public:
    template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U>& p) const {
        return std::hash<T>()(p.first) ^ std::hash<U>()(p.second);
    }
};

using BF16Cache = std::unordered_set<std::pair<std::size_t, std::size_t>, PairHash>;
using Weights = ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>;
using WeightsPtr = std::shared_ptr<Weights>;

// Design assessment: clearer type-level separation; export carries only write-side metadata.
struct ExportWeightsContext {
    bool is_weightless = false;
    std::unordered_map<const void*, std::size_t> const_to_offset;
};

// Import state stays separate; the descriptor codec still shares bidirectional control flow.
struct ImportWeightsContext {
    struct CtxHash {
        std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const {
            return (std::hash<std::size_t>()(p.first) + 0x9e3779b9) ^ (std::hash<std::size_t>()(p.second) + 0x9e3779b9);
        }
    };
    using ConstsCache = std::unordered_map<std::pair<std::size_t, std::size_t>, std::shared_ptr<ov::Node>, CtxHash>;

    bool is_weightless = false;
    WeightsPtr weights;
    std::string weights_path;
    ConstsCache consts_cache;
    BF16Cache bf16_consts;
    ov::FileHandleProvider handle_provider = nullptr;

    void reset();
};

}  // namespace s11n
}  // namespace npuw
}  // namespace ov
