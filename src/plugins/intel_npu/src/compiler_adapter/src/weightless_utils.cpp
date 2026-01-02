// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weightless_utils.hpp"

namespace intel_npu {

bool isInitMetadata(const NetworkMetadata& networkMetadata) {
    if (networkMetadata.inputs.size() == 0) {
        return false;
    }
    return networkMetadata.inputs.at(0).isInitInputWeights;
}

}  // namespace intel_npu
