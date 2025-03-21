// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

#include "openvino/core/visibility.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // Not implemented yet:
        R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModel.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canExportModel.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModelWithIncorrectConfig.*)",
        // requires export_model be implemented
        R"(.*Behavior.*OVCompiledModelBaseTest.*import_from_weightless_blob.*targetDevice=(MULTI|AUTO).*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*compile_from.*_blob.*targetDevice=(MULTI|AUTO).*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*use_blob_hint.*targetDevice=(MULTI|AUTO).*)",

        // unsupported metrics
        R"(.*smoke_AutoOVGetMetricPropsTest.*OVGetMetricPropsTest.*(AVAILABLE_DEVICES|OPTIMIZATION_CAPABILITIES|RANGE_FOR_ASYNC_INFER_REQUESTS|RANGE_FOR_STREAMS).*)",

        // Issue:
        // New API tensor tests
        R"(.*OVInferRequestCheckTensorPrecision.*type=i4.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*type=u1.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*type=u4.*)",

        // AUTO does not support import / export
        R"(.*smoke_Auto_BehaviorTests/OVCompiledGraphImportExportTest.*(mportExport|readFromV10IR).*/targetDevice=(AUTO).*)",
        R"(.*Behavior.*OVInferRequestIOTensorTest.*canInferAfterIOBlobReallocation.*)",
        R"(.*Behavior.*OVInferRequestDynamicTests.*InferUpperBoundNetworkAfterIOTensorsReshaping.*)",
        // template plugin doesn't support this case
        R"(.*OVInferRequestPerfCountersTest.*CheckOperationInProfilingInfo.*)"};

#if !defined(OPENVINO_ARCH_X86_64)
    // very time-consuming test
    retVector.emplace_back(R"(.*OVInferConsistencyTest.*)");
#endif
    return retVector;
}
