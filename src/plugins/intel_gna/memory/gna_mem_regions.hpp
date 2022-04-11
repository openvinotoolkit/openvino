// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

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

inline const char* rRegionToStr(const rRegion region) {
   const char* strRegion = "UNKNOWN";
   switch (region) {
        case REGION_INPUTS:
            strRegion = "REGION_INPUTS";
            break;
        case REGION_OUTPUTS:
            strRegion = "REGION_OUTPUTS";
            break;
        case REGION_SCRATCH:
            strRegion = "REGION_SCRATCH";
            break;
        case REGION_RO:
            strRegion = "REGION_RO";
            break;
        case REGION_STATES:
            strRegion = "REGION_STATES";
            break;
        case REGION_AUTO:
            strRegion = "REGION_AUTO";
            break;
   }
   return strRegion;
}

}  // namespace memory
}  // namespace GNAPluginNS
