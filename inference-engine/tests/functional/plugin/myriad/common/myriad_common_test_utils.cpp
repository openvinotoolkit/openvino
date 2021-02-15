// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_common_test_utils.hpp"

#include <cstdlib>
#include <string>

namespace CommonTestUtils {
namespace vpu {

bool CheckMyriad2() {
    if (const auto& envVar = std::getenv("IE_VPU_MYRIADX")) {
        return std::stoi(envVar) == 0;
    }
    return true;
}

}  // namespace vpu
}  // namespace CommonTestUtils

