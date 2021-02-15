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
    #if defined(_WIN32) || defined(_WIN64)
        // TODO: Issue 31197
        R"(.*(myriadLayersTestsProposal_smoke).*Caffe.*)",
        R"(.*(myriadLayersTestsProposal_smoke).*CaffeNoClipBeforeNms.*)",
        R"(.*(myriadLayersTestsProposal_smoke).*CaffeClipAfterNms.*)",
        R"(.*(myriadLayersTestsProposal_smoke).*CaffeNormalizedOutput.*)",
        R"(.*(myriadLayersTestsProposal_smoke).*TensorFlow.*)",
        R"(.*(myriadCTCDecoderLayerTests_smoke).*CTCGreedyDecoder.*)",
    #endif
    };
}
