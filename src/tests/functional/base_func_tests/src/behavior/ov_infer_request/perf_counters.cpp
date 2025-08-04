// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "behavior/ov_infer_request/perf_counters.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "common_test_utils/subgraph_builders/concat_with_params.hpp"

namespace ov {
namespace test {
namespace behavior {
void OVInferRequestPerfCountersTest::SetUp() {
    std::tie(target_device, configuration) = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
    function = ov::test::utils::make_concat_with_params();
    configuration.insert(ov::enable_profiling(true));
    execNet = core->compile_model(function, target_device, configuration);
    req = execNet.create_infer_request();
}

std::string OVInferRequestPerfCountersTest::getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
    std::string targetDevice;
    ov::AnyMap configuration;
    std::tie(targetDevice, configuration) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
    std::ostringstream result;
    result << "targetDevice=" << targetDevice << "_";
    if (!configuration.empty()) {
        using namespace ov::test::utils;
        for (auto &configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
        }
    }
    return result.str();
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

TEST_P(OVInferRequestPerfCountersExceptionTest, perfCountWereNotEnabledExceptionTest) {
    EXPECT_ANY_THROW(req.get_profiling_info());
}

TEST_P(OVInferRequestPerfCountersTest, CheckOperationInProfilingInfo) {
    req = execNet.create_infer_request();
    OV_ASSERT_NO_THROW(req.infer());

    std::vector<ov::ProfilingInfo> profiling_info;
    OV_ASSERT_NO_THROW(profiling_info = req.get_profiling_info());

    for (const auto& op : function->get_ops()) {
        if (!strcmp(op->get_type_info().name, "Constant"))
            continue;
        auto op_is_in_profiling_info = std::any_of(std::begin(profiling_info), std::end(profiling_info),
            [&] (const ov::ProfilingInfo& info) {
            if (info.node_name.find(op->get_friendly_name() + "_") != std::string::npos || info.node_name == op->get_friendly_name()) {
                return true;
            } else {
                return false;
            }
        });
        ASSERT_TRUE(op_is_in_profiling_info) << "For op: " << op;
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
