// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace GNAPluginNS {
namespace memory {

/**
 * @brief region of firmware data
 */
enum rRegion {
    REGION_INPUTS,
    REGION_OUTPUTS,
    REGION_SCRATCH,
    REGION_RO,
    REGION_STATES,
    REGION_AUTO
};

inline const char* rRegionToStr(uint8_t region) {
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
