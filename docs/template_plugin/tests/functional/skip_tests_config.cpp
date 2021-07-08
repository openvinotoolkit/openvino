// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*ExclusiveAsyncRequests.*",
        ".*ReusableCPUStreamsExecutor.*",
        R"(.*SplitLayerTest.*numSplits\=30.*)",
        // CVS-51758
        ".*InferRequestPreprocessConversionTest.*oLT=NHWC.*",
        ".*InferRequestPreprocessDynamicallyInSetBlobTest.*oPRC=0.*oLT=1.*",
    };
}
