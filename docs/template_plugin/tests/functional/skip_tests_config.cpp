// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        R"(.*ExclusiveAsyncRequests.*)",
        R"(.*ReusableCPUStreamsExecutor.*)",
        R"(.*SplitLayerTest.*numSplits\=30.*)",
        // CVS-51758
        R"(.*InferRequestPreprocessConversionTest.*oLT=(NHWC|NCHW).*)",
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*oPRC=0.*oLT=1.*)",
        // CVS-58963: Not implemented yet
        R"(.*Behavior.*InferRequest.*OutOfFirstOutIsInputForSecondNetwork.*)",
    };
}
