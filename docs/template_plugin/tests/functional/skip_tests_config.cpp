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
        //Not Implemented
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(canSetConfigToExecNet|canSetConfigToExecNetAndCheckConfigAndCheck).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CheckExecGraphInfoBeforeExecution|CheckExecGraphInfoAfterExecution|CheckExecGraphInfoSerialization).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CanCreateTwoExeNetworksAndCheckFunction).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(checkGetExecGraphInfoIsNotNullptr).*)",
        R"(.*smoke_BehaviorTests.*OVExecNetwork.ieImportExportedFunction.*)",

        // TODO: Round with f16 is not supported
        R"(.*smoke_Hetero_BehaviorTests.*OVExecNetwork.*readFromV10IR.*)",
        // TODO: execution graph is not supported
        R"(.*ExecGraph.*)",

        // TODO: support import / export of precisions in template plugin
        R"(.*smoke_Hetero_BehaviorTests.*OVExecNetwork.ieImportExportedFunction.*)",
        R"(.*smoke_BehaviorTests.*OVExecNetwork.ieImportExportedFunction.*)",

        // TODO: Round with f16 is not supported
        R"(.*smoke_Hetero_BehaviorTests.*OVExecNetwork.*readFromV10IR.*)",

        // CVS-64094
        R"(.*ReferenceLogSoftmaxLayerTest.*4.*iType=f16.*axis=.*1.*)",
        // CVS-64080
        R"(.*ReferenceMishLayerTest.*dimensionDynamic.*)",
        //CVS-64012
        R"(.*ReferenceDeformableConvolutionLayerTest.*f16.*real_offset_padding_stride_dialation.*)",
        R"(.*ReferenceDeformableConvolutionLayerTest.*bf16.*)",
        R"(.*ReferenceDeformableConvolutionV8LayerTest.*f16.*real_offset_padding_stride_dialation.*)",
        R"(.*ReferenceDeformableConvolutionV8LayerTest.*bf16.*)",
    };

#ifdef _WIN32
    // CVS-63989
     retVector.emplace_back(R"(.*ReferenceSigmoidLayerTest.*u64.*)");
#endif
    return retVector;
}
