// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "cpu_memory.h"
#include "executor_config.hpp"

namespace ov {
namespace intel_cpu {

enum MVNLayoutType {
    mvn_planar,
    mvn_block,
    mvn_by_channel
};

// Defines way to add epsilon: inside sqrt or outside.
enum MVNEpsMode {
    INSIDE_SQRT,
    OUTSIDE_SQRT
};

struct MVNAttrs {
    MVNLayoutType layout = mvn_planar;
    bool initAcrossChannels_ = false;
    bool execAcrossChannels_ = false;
    bool normalizeVariance_ = false;
    float epsValue_ = 0.0f;
    MVNEpsMode epsMode_ = INSIDE_SQRT;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    bool srcIsNHWC = false;
    std::vector<const void*> postOpsDataPtrs;
    VectorDims shape5D = {0, 0, 0, 0, 0};
};

using MVNConfig = executor::Config<MVNAttrs>;

}   // namespace intel_cpu
}   // namespace ov