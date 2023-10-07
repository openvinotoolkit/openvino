// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <ie_system_conf.h>

#include <string>
#include <vector>

#include "ie_parallel.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // TODO: Issue: 43793
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*iPRC=0.*_iLT=1.*)",
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*oPRC=0.*_oLT=1.*)",

        // Not expected behavior
        R"(.*Behavior.*InferRequestSetBlobByType.*Batched.*)",
        R"(.*Auto.*Behavior.*ExecutableNetworkBaseTest.*canLoadCorrectNetworkToGetExecutableWithIncorrectConfig.*)",

        // Not implemented yet:
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModel.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canExportModel.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNetWithIncorrectConfig.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModelWithIncorrectConfig.*)",

        // TODO: CVS-104942
        R"(.*(Auto|Multi).*Behavior.*ExecutableNetworkBaseTest.*canLoadCorrectNetworkToGetExecutableAndCheckConfig.*)",
        R"(.*(Auto|Multi).*SetPropLoadNetWorkGetPropTests.*)",

        // CPU does not support dynamic rank
        // Issue: CVS-66778
        R"(.*smoke_Auto_BehaviorTests.*InferFullyDynamicNetworkWith(S|G)etTensor.*)",
        R"(.*smoke_Auto_BehaviorTests.*DynamicOutputToDynamicInput.*)",
        R"(.*smoke_Auto_BehaviorTests.*DynamicInputToDynamicOutput.*)",
        // unsupported metrics
        R"(.*smoke_AutoOVGetMetricPropsTest.*OVGetMetricPropsTest.*(AVAILABLE_DEVICES|OPTIMIZATION_CAPABILITIES|RANGE_FOR_ASYNC_INFER_REQUESTS|RANGE_FOR_STREAMS).*)",

        // Issue:
        // New API tensor tests
        R"(.*OVInferRequestCheckTensorPrecision.*type=i4.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*type=u1.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*type=u4.*)",

        // AUTO does not support import / export
        R"(.*smoke_Auto_BehaviorTests/OVCompiledGraphImportExportTest.*(mportExport|readFromV10IR).*/targetDevice=(AUTO).*)",

        // New plugin API doesn't support changes of pre-processing
        R"(.*(Auto|Multi).*InferRequestPreprocessTest.*SetPreProcessToInputInfo.*)",
        R"(.*(Auto|Multi).*InferRequestPreprocessTest.*SetPreProcessToInferRequest.*)",
        // New plugin work with tensors, so it means that blob in old API can have different pointers
        R"(.*(Auto|Multi).*InferRequestIOBBlobTest.*secondCallGetInputDoNotReAllocateData.*)",
        R"(.*(Auto|Multi).*InferRequestIOBBlobTest.*secondCallGetOutputDoNotReAllocateData.*)",
        R"(.*(Auto|Multi).*InferRequestIOBBlobTest.*secondCallGetInputAfterInferSync.*)",
        R"(.*(Auto|Multi).*InferRequestIOBBlobTest.*secondCallGetOutputAfterInferSync.*)",
        // TODO Issue 100145
        R"(.*Behavior.*InferRequestIOBBlobTest.*canReallocateExternalBlobViaGet.*)",
        R"(.*Behavior.*OVInferRequestIOTensorTest.*canInferAfterIOBlobReallocation.*)",
        R"(.*Behavior.*OVInferRequestDynamicTests.*InferUpperBoundNetworkAfterIOTensorsReshaping.*)",
        // Not expected behavior
        R"(.*Behavior.*(Multi|Auto).*InferRequestSetBlobByType.*Batched.*)",
        R"(.*(Multi|Auto).*Behavior.*InferRequestIOBBlobTest.*canProcessDeallocatedOutputBlobAfterGetAndSetBlob.*)",
        // template plugin doesn't support this case
        R"(.*OVInferRequestPerfCountersTest.*CheckOperationInProfilingInfo.*)"};

#if !defined(OPENVINO_ARCH_X86_64)
    // very time-consuming test
    retVector.emplace_back(R"(.*OVInferConsistencyTest.*)");
#endif

#if defined(_WIN32)
    retVector.emplace_back(R"(.*LoadNetworkCompiledKernelsCacheTest.*CanCreateCacheDirAndDumpBinariesUnicodePath.*)");
#endif
    return retVector;
}
