// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_types.h"
#include "executor_config.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

/**
 * @brief MVN layout types
 * - mvn_planar: NCDHW/NCHW format (channels as separate planes)
 * - mvn_block: Blocked format (nCsp8c, nCsp16c)
 * - mvn_by_channel: NDHWC/NHWC format (channels as innermost dimension)
 */
enum MVNLayoutType : std::uint8_t { mvn_planar, mvn_block, mvn_by_channel };

/**
 * @brief MVN epsilon mode
 * - INSIDE_SQRT: epsilon is added inside square root: sqrt(variance + eps)
 * - OUTSIDE_SQRT: epsilon is added outside square root: sqrt(variance) + eps
 */
enum MVNEpsMode : std::uint8_t { INSIDE_SQRT, OUTSIDE_SQRT };

/**
 * @brief MVN operation attributes
 *
 * This structure contains all configuration parameters for MVN operation.
 * It supports different normalization modes, layouts, and precisions.
 */
struct MVNAttrs {
    MVNLayoutType layout = mvn_planar;
    bool initAcrossChannels_ = false;
    bool execAcrossChannels_ = false;
    bool normalizeVariance_ = true;
    float epsValue_ = 1e-9F;
    MVNEpsMode epsMode_ = INSIDE_SQRT;
    VectorDims shape5D;
    size_t actualChannelSize = 0;  // Actual channel size for post-ops
    PostOps postOps;               // Post-operations configuration
};

using MVNConfig = executor::Config<MVNAttrs>;

}  // namespace ov::intel_cpu
