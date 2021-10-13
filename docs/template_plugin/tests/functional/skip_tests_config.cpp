// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // CVS-66280
        R"(.*canLoadCorrectNetworkAndCheckConfig.*)",
        R"(.*canSetCorrectConfigLoadNetworkAndCheckConfig.*)",
        //
        R"(.*ExclusiveAsyncRequests.*)",
        R"(.*ReusableCPUStreamsExecutor.*)",
        R"(.*SplitLayerTest.*numSplits=30.*)",
        // CVS-51758
        R"(.*InferRequestPreprocessConversionTest.*oLT=(NHWC|NCHW).*)",
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*oPRC=0.*oLT=1.*)",
        // CVS-64094
        R"(.*ReferenceLogSoftmaxLayerTest.*4.*iType=f16.*axis=.*1.*)",
        // CVS-64080
        R"(.*ReferenceMishLayerTest.*dimensionDynamic.*)"
    };

#ifdef _WIN32
    // CVS-63989
     retVector.emplace_back(R"(.*ReferenceSigmoidLayerTest.*u64.*)");
#endif
    return retVector;
}
