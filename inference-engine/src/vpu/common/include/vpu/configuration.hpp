// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>

#include "caseless.hpp"

#include "vpu/utils/optional.hpp"

namespace vpu {

struct CompilationConfig {
    //
    // Deprecated options
    //

    float inputScale = 1.0f;
    float inputBias = 0.0f;
};

}  // namespace vpu
