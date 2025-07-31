// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>
#include <ostream>
#include <vector>

#include "cpu_types.h"
#include "executor_config.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

enum class EltwiseImplType : uint8_t { reference = 0, optimized = 1, optimizedShapeAgnostic = 2 };

enum class EltwiseBroadcastingPolicy : uint8_t {
    PerChannel,
    PerTensor,
    Undefined,
};

struct EltwiseData {
    Algorithm algo = Algorithm::Default;
    dnnl::algorithm onednnAlgorithm = dnnl::algorithm::undef;
    float alpha = 0.0F;
    float beta = 0.0F;
    float gamma = 0.0F;

    bool operator==(const EltwiseData& rhs) const noexcept {
        return algo == rhs.algo && onednnAlgorithm == rhs.onednnAlgorithm && alpha == rhs.alpha && beta == rhs.beta &&
               gamma == rhs.gamma;
    }
};

inline std::ostream& operator<<(std::ostream& os, const EltwiseData& eltwiseData) {
    os << "EltwiseData(algo: " << algToString(eltwiseData.algo)
       << ", onednnAlgorithm: " << static_cast<int>(eltwiseData.onednnAlgorithm) << ", alpha: " << eltwiseData.alpha
       << ", beta: " << eltwiseData.beta << ", gamma: " << eltwiseData.gamma << ")";
    return os;
}

struct EltwiseAttrs {
    EltwiseData data;

    std::vector<ptrdiff_t> start_offset_in;
    ptrdiff_t start_offset_out = 0;

    EltwiseBroadcastingPolicy broadcastingPolicy = EltwiseBroadcastingPolicy::Undefined;

    std::vector<float> scales;
    std::vector<float> shifts;

    // For fused operations
    std::vector<EltwiseData> fusedEltwiseData;
    std::vector<Type> opsList;

    bool specialConvolutionAddFusing = false;

    PostOps postOps;
};

using EltwiseConfig = executor::Config<EltwiseAttrs>;

}  // namespace ov::intel_cpu
