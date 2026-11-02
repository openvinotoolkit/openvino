// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "quant/weights.hpp"

namespace ov_gguf_test {

using UnpackedGGUFLoad = std::tuple<std::unordered_map<std::string, ov::frontend::gguf::GGUFMetaData>,
                                    std::unordered_map<std::string, ov::Tensor>,
                                    std::unordered_map<std::string, ov::frontend::gguf::GgufTensorType>,
                                    std::vector<std::shared_ptr<ov::MappedMemory>>>;

inline UnpackedGGUFLoad unpack_gguf_load(ov::frontend::gguf::GGUFLoad loaded) {
    auto& [metadata, raw_weights, qtypes, mappings] = loaded;
    std::unordered_map<std::string, ov::Tensor> weights;
    for (const auto& [name, raw] : raw_weights) {
        auto tensors = ov::frontend::gguf::unpack_gguf_weight(raw);
        weights.emplace(name, std::move(tensors.weight));
        const auto prefix = name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0
                                ? name.substr(0, name.size() - 7)
                                : name;
        if (tensors.scales)
            weights.emplace(prefix + ".scales", std::move(tensors.scales));
        if (tensors.zero_point)
            weights.emplace(prefix + ".zp", std::move(tensors.zero_point));
    }
    return {std::move(metadata), std::move(weights), std::move(qtypes), std::move(mappings)};
}

inline UnpackedGGUFLoad get_unpacked_gguf_data(const std::filesystem::path& file) {
    return unpack_gguf_load(ov::frontend::gguf::get_gguf_data(file));
}

inline UnpackedGGUFLoad get_unpacked_gguf_data(const std::vector<std::filesystem::path>& files) {
    return unpack_gguf_load(ov::frontend::gguf::get_gguf_data(files));
}

}  // namespace ov_gguf_test
