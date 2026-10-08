// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_npu/common/igraph.hpp"

namespace intel_npu {

namespace zeroProfiling {
struct ProfilingQuery;
}

/**
 * @brief Converts a pipeline's compiler-specific profiling data into OpenVINO profiling records.
 */
class IProfilingDecoder {
public:
    virtual std::vector<ov::ProfilingInfo> decode(const IGraph& graph,
                                                  const zeroProfiling::ProfilingQuery& query) const = 0;

    virtual ~IProfilingDecoder() = default;
};

}  // namespace intel_npu
