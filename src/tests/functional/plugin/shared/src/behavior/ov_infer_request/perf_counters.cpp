// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "behavior/ov_infer_request/perf_counters.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestPerfCountersTest::getTestCaseName(const testing::TestParamInfo<InferRequestParams>& obj) {
    return OVInferRequestTests::getTestCaseName(obj);
}

void OVInferRequestPerfCountersTest::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::tie(targetDevice, configuration) = this->GetParam();
    function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(targetDevice);
    configuration.insert(ov::enable_profiling(true));
    execNet = core->compile_model(function, targetDevice, configuration);
    req = execNet.create_infer_request();
}

TEST_P(OVInferRequestPerfCountersTest, NotEmptyAfterAsyncInfer) {
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    std::vector<ov::ProfilingInfo> perf;
    OV_ASSERT_NO_THROW(perf = req.get_profiling_info());
    ASSERT_FALSE(perf.empty());
}

TEST_P(OVInferRequestPerfCountersTest, NotEmptyAfterSyncInfer) {
    OV_ASSERT_NO_THROW(req.infer());
    std::vector<ov::ProfilingInfo> perf;
    OV_ASSERT_NO_THROW(perf = req.get_profiling_info());
    ASSERT_FALSE(perf.empty());
}
}  // namespace behavior
}  // namespace test
}  // namespace ov