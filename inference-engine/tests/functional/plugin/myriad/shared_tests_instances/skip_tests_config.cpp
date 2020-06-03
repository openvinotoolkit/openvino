// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    // Not supported activation types
    std::vector<std::string> unsupportedActivationTypes = {
        ".*ActivationLayerTest\\.CompareWithRefs/Tanh.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Exp.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Log.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Sigmoid.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Relu.*netPRC=FP32.*"
    };

    std::vector<std::string> behaviorTests = {
        ".*Behavior.*ExecGraphTests.*"
    };

    // Issue 26268
    std::vector<std::string> issue26268 = {
        ".*ConcatLayerTest.*axis=0.*"
    };

    std::vector<std::string> testsToDisable;
    testsToDisable.insert(testsToDisable.end(), unsupportedActivationTypes.begin(), unsupportedActivationTypes.end());
    testsToDisable.insert(testsToDisable.end(), behaviorTests.begin(), behaviorTests.end());
    testsToDisable.insert(testsToDisable.end(), issue26268.begin(), issue26268.end());

    #if defined(_WIN32) || defined(WIN32)
        // Issue 31197
        std::vector<std::string> issue31197 = {
            ".*IEClassBasicTestP\\.smoke_registerPluginsXMLUnicodePath/0.*",
            ".*myriadLayersTestsProposal_smoke\\.Caffe.*",
            ".*myriadLayersTestsProposal_smoke\\.CaffeNoClipBeforeNms.*",
            ".*myriadLayersTestsProposal_smoke\\.CaffeClipAfterNms.*",
            ".*myriadLayersTestsProposal_smoke\\.CaffeNormalizedOutput.*",
            ".*myriadLayersTestsProposal_smoke\\.TensorFlow.*",
            ".*myriadCTCDecoderLayerTests_smoke\\.CTCGreedyDecoder/0.*",
            ".*myriadCTCDecoderLayerTests_smoke\\.CTCGreedyDecoder/1.*",
            ".*myriadCTCDecoderLayerTests_smoke\\.CTCGreedyDecoder/2.*",
            ".*myriadCTCDecoderLayerTests_smoke\\.CTCGreedyDecoder/3.*"
        };

        testsToDisable.insert(testsToDisable.end(), issue31197.begin(), issue31197.end());
    #endif

    return testsToDisable;
}
