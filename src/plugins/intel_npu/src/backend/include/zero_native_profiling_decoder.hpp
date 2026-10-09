// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "zero_profiling.hpp"
#include "zero_profiling_decoder.hpp"

namespace intel_npu {

/**
 * @brief Decodes profiling data the driver has already resolved into layer statistics; no further processing needed.
 */
class NativeProfilingDecoder final : public IProfilingDecoder {
public:
    std::vector<ov::ProfilingInfo> decode(const IGraph&, const zeroProfiling::ProfilingQuery& query) const override {
        return query.getLayerStatistics();
    }
};

}  // namespace intel_npu
