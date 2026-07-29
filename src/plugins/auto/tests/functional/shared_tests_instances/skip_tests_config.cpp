// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

#include "openvino/core/visibility.hpp"

const std::vector<std::regex>& disabled_test_patterns() {
    const static std::vector<std::regex> patterns{
        // Not implemented yet:
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModel.*)"),
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*canExportModel.*)"),
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModelWithIncorrectConfig.*)"),
        // requires export_model be implemented
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*import_from_weightless_blob.*targetDevice=(MULTI|AUTO).*)"),
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*compile_from.*_blob.*targetDevice=(MULTI|AUTO).*)"),
        std::regex(R"(.*Behavior.*OVCompiledModelBaseTest.*use_blob_hint.*targetDevice=(MULTI|AUTO).*)"),

        // unsupported metrics
        std::regex(
            R"(.*smoke_AutoOVGetMetricPropsTest.*OVGetMetricPropsTest.*(AVAILABLE_DEVICES|OPTIMIZATION_CAPABILITIES|RANGE_FOR_ASYNC_INFER_REQUESTS|RANGE_FOR_STREAMS).*)"),

        // Issue:
        // New API tensor tests
        std::regex(R"(.*OVInferRequestCheckTensorPrecision.*type=i4.*)"),
        std::regex(R"(.*OVInferRequestCheckTensorPrecision.*type=u1.*)"),
        std::regex(R"(.*OVInferRequestCheckTensorPrecision.*type=u4.*)"),

        // AUTO does not support import / export
        std::regex(
            R"(.*smoke_Auto_BehaviorTests/OVCompiledGraphImportExportTest.*(mportExport|readFromV10IR).*/targetDevice=(AUTO).*)"),
        std::regex(R"(.*Behavior.*OVInferRequestIOTensorTest.*canInferAfterIOBlobReallocation.*)"),
        std::regex(R"(.*Behavior.*OVInferRequestDynamicTests.*InferUpperBoundNetworkAfterIOTensorsReshaping.*)"),
        // template plugin doesn't support this case
        std::regex(R"(.*OVInferRequestPerfCountersTest.*CheckOperationInProfilingInfo.*)"),
        // model import is not supported
        std::regex(R"(.*OVCompiledModelBaseTest.import_from_.*)"),

#if !defined(OPENVINO_ARCH_X86_64)
        // very time-consuming test
        std::regex(R"(.*OVInferConsistencyTest.*)"),
#endif
    };

    return patterns;
}
