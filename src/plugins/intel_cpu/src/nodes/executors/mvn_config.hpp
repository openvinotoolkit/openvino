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

enum MVNLayoutType : std::uint8_t { mvn_planar, mvn_block, mvn_by_channel };

enum MVNEpsMode : std::uint8_t { INSIDE_SQRT, OUTSIDE_SQRT };

struct MVNAttrs {
    MVNLayoutType layout = mvn_planar;
    bool initAcrossChannels_ = false;
    bool execAcrossChannels_ = false;
    bool normalizeVariance_ = true;
    float epsValue_ = 1e-9f;
    MVNEpsMode epsMode_ = INSIDE_SQRT;
    ov::element::Type src_prc = ov::element::f32;
    ov::element::Type dst_prc = ov::element::f32;
    VectorDims shape5D;
    PostOps postOps;
};

using MVNConfig = executor::Config<MVNAttrs>;

}  // namespace ov::intel_cpu