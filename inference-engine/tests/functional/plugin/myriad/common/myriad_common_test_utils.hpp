// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <iostream>

namespace CommonTestUtils {
namespace vpu {

bool CheckMyriadX() {
    if (auto envVar = std::getenv("IE_VPU_MYRIADX")) {
        return std::stoi(envVar) != 0;
    }
    return false;
}

}  // namespace vpu
}  // namespace CommonTestUtils
