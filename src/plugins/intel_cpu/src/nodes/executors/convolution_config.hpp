// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "executor_config.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

enum class ZeroPointsType : uint8_t { None, PerTensor, PerChannel };
enum class AutoPaddingType : uint8_t { None, SAME_UPPER, SAME_LOWER };
/**
 * @todo only attributes necessary for 1x1 convlution as fullyconnected fallback
 * are currently listed
 */
struct ConvAttrs {
    std::vector<size_t> stride;
    std::vector<size_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    AutoPaddingType autoPadding = AutoPaddingType::None;

    bool withBias = false;
    bool weightsNonTransposed = false;
    bool isGrouped = false;
    // @todo can we just check for port precisions instead?
    bool isGraphQuantized = false;
    bool fcSemantic = false;
    bool nonConstantWeights = false;
    ZeroPointsType inputZeroPointsType = ZeroPointsType::None;
    std::vector<float> dqScales;

    PostOps postOps;
};

using ConvConfig = executor::Config<ConvAttrs>;

}  // namespace ov::intel_cpu
