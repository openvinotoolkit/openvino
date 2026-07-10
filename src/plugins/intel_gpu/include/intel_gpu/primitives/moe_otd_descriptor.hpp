// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <filesystem>
#include <string>
#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief Lightweight, serializable descriptor for the offload-to-disk (OTD) weight strategy.
/// @details Consolidates the OTD plumbing so the primitive carries a single cohesive field
/// instead of loose members. Empty/zero values mean OTD is disabled (fully resident). This is
/// the only OTD state that participates in blob-cache serialization; the runtime
/// IExpertWeightProvider is rebuilt from it on the impl side.
struct moe_otd_descriptor {
    std::vector<size_t> weight_bin_offsets;
    std::filesystem::path weights_path;
    size_t lru_expert_num = 0;

    bool operator==(const moe_otd_descriptor& rhs) const {
        return weight_bin_offsets == rhs.weight_bin_offsets && weights_path == rhs.weights_path &&
               lru_expert_num == rhs.lru_expert_num;
    }

    void save(BinaryOutputBuffer& ob) const {
        ob << weight_bin_offsets;
        ob << weights_path.string();
        ob << lru_expert_num;
    }

    void load(BinaryInputBuffer& ib) {
        ib >> weight_bin_offsets;
        std::string path_str;
        ib >> path_str;
        weights_path = path_str;
        ib >> lru_expert_num;
    }
};

}  // namespace cldnn
