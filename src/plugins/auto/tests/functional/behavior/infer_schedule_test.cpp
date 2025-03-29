// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"

namespace ov {
namespace auto_plugin {
namespace tests {
using schedule_policy_param = std::tuple<ov::AnyMap,  // properties with schedule policy setting
                                         int          // number of created infer requests
                                         >;

class InferSchedulePolicyTest : public AutoFuncTests, public testing::WithParamInterface<schedule_policy_param> {
public:
    void SetUp() override {
        AutoFuncTests::SetUp();
        std::tie(property, niters) = this->GetParam();
    }
    static std::string getTestCaseName(const testing::TestParamInfo<schedule_policy_param>& obj) {
        ov::AnyMap property;
        int niters;
        std::tie(property, niters) = obj.param;
        std::ostringstream result;
        result << "numberOfInfer=" << niters << "_";
        if (!property.empty()) {
            for (auto& iter : property) {
                result << "priority=" << iter.first << "_" << iter.second.as<std::string>();
            }
        }
        return result.str();
    }

public:
    ov::AnyMap property;
    int niters;
};

TEST_P(InferSchedulePolicyTest, can_run_async_requests_with_different_schedule_policy) {
    ov::CompiledModel compiled_model;
    property.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model_cannot_batch, "AUTO", property));
    std::vector<ov::InferRequest> inferReqsQueue;
    int count = niters;
    while (count--) {
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());
        inferReqsQueue.push_back(req);
    }
    for (auto& req : inferReqsQueue) {
        OV_ASSERT_NO_THROW(req.start_async());
    }
    for (auto& req : inferReqsQueue) {
        OV_ASSERT_NO_THROW(req.wait());
    }
}

TEST_P(InferSchedulePolicyTest, can_run_sync_requests_with_different_schedule_policy) {
    ov::CompiledModel compiled_model;
    property.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model_cannot_batch, "AUTO", property));
    std::vector<ov::InferRequest> inferReqsQueue;
    int count = niters;
    while (count--) {
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());
        inferReqsQueue.push_back(req);
    }
    for (auto& req : inferReqsQueue) {
        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(req.wait());
    }
}

auto properties = std::vector<ov::AnyMap>{
    {ov::device::priorities("MOCK_GPU"), ov::intel_auto::schedule_policy(ov::intel_auto::SchedulePolicy::ROUND_ROBIN)},
    {ov::device::priorities("MOCK_GPU"),
     ov::intel_auto::schedule_policy(ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY)},
    {ov::device::priorities("MOCK_CPU"), ov::intel_auto::schedule_policy(ov::intel_auto::SchedulePolicy::ROUND_ROBIN)},
    {ov::device::priorities("MOCK_CPU"),
     ov::intel_auto::schedule_policy(ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY)},
    {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
     ov::intel_auto::schedule_policy(ov::intel_auto::SchedulePolicy::ROUND_ROBIN)},
    {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
     ov::intel_auto::schedule_policy(ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY)},
    {ov::device::priorities("MOCK_CPU", "MOCK_GPU"),
     ov::intel_auto::schedule_policy(ov::intel_auto::SchedulePolicy::ROUND_ROBIN)}};
auto niters = std::vector<int>{10, 20, 30};

INSTANTIATE_TEST_SUITE_P(AutoFuncTests,
                         InferSchedulePolicyTest,
                         ::testing::Combine(::testing::ValuesIn(properties), ::testing::ValuesIn(niters)),
                         InferSchedulePolicyTest::getTestCaseName);
}  // namespace tests
}  // namespace auto_plugin
}  // namespace ov
