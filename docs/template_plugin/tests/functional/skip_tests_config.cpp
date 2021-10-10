// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
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

        // TODO: execution graph is not supported
        R"(.*ExecGraph.*)",

        // Multi / Auto don't support Import / Export
        R"(.*smoke_(Auto|Multi)_BehaviorTests.*OVExecNetwork.*importExportedNetwork.*)",
        R"(.*smoke_(Auto|Multi)_BehaviorTests.*OVExecNetwork.*importExportedIENetwork.*)",
        R"(.*smoke_(Auto|Multi)_BehaviorTests.*OVExecNetwork.*readFromV10IR.*)",

        // TODO: Round with f16 is not supported
        R"(.*smoke_Hetero_BehaviorTests.*OVExecNetwork.*readFromV10IR.*)",
    };
}
