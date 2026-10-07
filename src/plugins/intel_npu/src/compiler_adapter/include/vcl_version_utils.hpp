// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>

#include "openvino/core/except.hpp"

namespace intel_npu::vcl_version_utils {

/**
 * @brief The VCL API version the plugin will actually speak: the lower of what the plugin was built
 * against and what the loaded compiler library reports.
 */
struct UsedVersion {
    uint16_t Major;
    uint16_t Minor;
};

/**
 * @brief Negotiates the VCL version to use.
 *
 * Expressed in plain integers rather than `vcl_version_info_t`, since the negotiation itself has no
 * dependency on VCL types.
 *
 * - Same major: take the lower minor.
 * - Plugin major newer than the library: downgrade to the library's version.
 * - Plugin major older than the library: keep the plugin's version.
 */
inline UsedVersion getUsedVclVersion(uint16_t pluginMajor,
                                     uint16_t pluginMinor,
                                     uint16_t loadedMajor,
                                     uint16_t loadedMinor) {
    uint16_t usedMajor = pluginMajor, usedMinor = pluginMinor;
    if (pluginMajor == loadedMajor) {
        usedMinor = std::min(pluginMinor, loadedMinor);
    } else if (pluginMajor > loadedMajor) {
        usedMajor = loadedMajor;
        usedMinor = loadedMinor;
    }
    return {usedMajor, usedMinor};
}

/**
 * @brief Throws if the negotiated version is below the floor the plugin requires.
 * @param usedVersion Result of `getUsedVclVersion`.
 * @param loadedMajor,loadedMinor Version reported by the compiler library, used for the message.
 * @param floorMajor,floorMinor Minimum version the plugin supports (`VCL_COMPILER_VERSION_*`).
 */
inline void checkVclVersion(const UsedVersion& usedVersion,
                            uint16_t loadedMajor,
                            uint16_t loadedMinor,
                            uint16_t floorMajor,
                            uint16_t floorMinor) {
    if (usedVersion.Major < floorMajor || (usedVersion.Major == floorMajor && usedVersion.Minor < floorMinor)) {
        OPENVINO_THROW("Unsupported VCL version: ",
                       loadedMajor,
                       ".",
                       loadedMinor,
                       ", please use VCL ",
                       floorMajor,
                       ".",
                       floorMinor,
                       " or later");
    }
}

}  // namespace intel_npu::vcl_version_utils
