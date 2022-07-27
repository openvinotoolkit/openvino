// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

namespace GNAPluginNS {
namespace memory {

/**
 * @brief Logical region of model memory.
 * Needed for models for embedded GNA
 * When model is exported for non-embedded uses its memory is exported following the enum value order
 */
enum rRegion {
    REGION_INPUTS = 0x0,
    REGION_OUTPUTS = 0x1,
    REGION_SCRATCH = 0x10,
    REGION_STATES = 0x100,
    REGION_RO = 0x1000,
    REGION_AUTO = 0x10000,
};

inline std::map<rRegion, std::string> GetAllRegionsToStrMap() {
    return {
        {REGION_INPUTS,  "REGION_INPUTS"},
        {REGION_OUTPUTS, "REGION_OUTPUTS"},
        {REGION_SCRATCH, "REGION_SCRATCH"},
        {REGION_STATES,  "REGION_STATES"},
        {REGION_RO,      "REGION_RO"},
        {REGION_AUTO,    "REGION_AUTO"}
    };
}

inline std::string rRegionToStr(const rRegion region) {
    const auto& map = GetAllRegionsToStrMap();
    const auto found = map.find(region);
    if (found == map.end()) {
        return "UNKNOWN";
    }
    return found->second;
}

}  // namespace memory
}  // namespace GNAPluginNS
