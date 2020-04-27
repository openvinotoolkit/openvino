// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_tests_config.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include <gtest/gtest.h>

namespace vpu {
namespace tests {

const char* pluginName      () { return "myriadPlugin"; }
const char* pluginNameShort () { return "myriad"; }
const char* deviceName      () { return "MYRIAD"; }
bool        deviceForceReset() { return true; }

}  // namespace tests
}  // namespace vpu

std::vector<std::string> disabledTestPatterns() {
    return {
    };
}