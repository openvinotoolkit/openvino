// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"

namespace ov::intel_cpu {

/**
 * @todo only attributes necessary for 1x1 convlution as fullyconnected fallback
 * are currently listed
 */
struct ConvAttrs {
    bool withBias;
};

using ConvConfig = executor::Config<ConvAttrs>;

}  // namespace ov::intel_cpu
