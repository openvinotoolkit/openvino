// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {

/**
 * @brief Decodes raw device profiling data into OpenVINO profiling records.
 * @details Expressed only in OpenVINO and standard types so callers (e.g. Graph) do not need to know which
 * backend produced the decoding (VCL compiler library or driver).
 */
class IProfilingDecoder {
public:
    /**
     * @param profData The raw profiling buffer collected from the device for one inference.
     * @param network The compiled network binary the profiling data refers to.
     * @return The decoded, per-layer profiling information.
     */
    virtual std::vector<ov::ProfilingInfo> decode(const std::vector<uint8_t>& profData,
                                                  const std::vector<uint8_t>& network) const = 0;

    virtual ~IProfilingDecoder() = default;
};

}  // namespace intel_npu
