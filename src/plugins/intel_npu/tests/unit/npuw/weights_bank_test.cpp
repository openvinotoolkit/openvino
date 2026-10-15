// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "fake_npu_remote_context.hpp"
#include "lazy_tensor.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"
#include "weights_bank.hpp"

namespace {

using ov::test::npuw::FakeNpuRemoteContext;
using ov::test::npuw::tensor_data_ptr;

std::vector<float> read_f32(const ov::Tensor& tensor) {
    ov::Tensor host(tensor.get_element_type(), tensor.get_shape());
    tensor.copy_to(host);
    return {host.data<float>(), host.data<float>() + host.get_size()};
}

class WeightsBankTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        m_core = std::make_shared<testing::NiceMock<ov::MockICore>>();
        m_remote_context = std::make_shared<FakeNpuRemoteContext>();
        ON_CALL(*m_core, get_default_context(testing::_))
            .WillByDefault(testing::Return(ov::SoPtr<ov::IRemoteContext>{m_remote_context, nullptr}));
    }

    std::shared_ptr<testing::NiceMock<ov::MockICore>> m_core;
    std::shared_ptr<FakeNpuRemoteContext> m_remote_context;
};

}  // namespace

TEST_P(WeightsBankTest, EvaluationUpdatesRuntimeWeightSources) {
    using ov::npuw::weights::Bank;
    using ov::npuw::weights::LazyTensor;

    const auto& device = GetParam();
    auto shared_context = std::make_shared<ov::weight_sharing::Context>();
    Bank bank_instance(m_core, device, "first-bank", shared_context);

    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;
    constants.push_back(std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                               ov::Shape{2},
                                                               std::vector<float>{1.0f, 2.0f}));
    constants.push_back(std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                         ov::Shape{2},
                                                         std::vector<float>{3.0f, 4.0f}));
    // Collect LazyTensor UIDs:
    // either a shared "constant" or a shared "transformation" of such constants.
    // At this point, NPUW doesn't discern shared "constant" and shared "transformation" of constants.
    // Let's interpret it as "shared transformation"(LazyTensor) as the broader concept.
    std::set<int64_t> uids;
    for (const auto& constant : constants) {
        uids.insert(bank_instance.registerLT(LazyTensor(constant), device));
    }

    // Evaluate and allocate memory for the "shared transformation" and register it in ov::weight_sharing::Context.
    // The memory for the "shared transformation" is allocated and registered in the shared context 
    // as "source_id" that equals to the UID of the "shared transformation",
    // "constant_id" is 0.
    auto statBeforeEvaluate = this->m_remote_context->get_statistics_snapshot();
    bank_instance.evaluate_and_allocate();

    // REQUIREMENT: "source_id" from ov::weight_sharing::Context must match the UID of "shared transformation"
    ASSERT_EQ(shared_context->m_runtime_sources.size(), uids.size());
    const auto sources_after_first_evaluate = shared_context->m_runtime_sources;
    for (const auto& [source_id, source] : sources_after_first_evaluate) {
        // REQUIREMENT reinforcement: The source_id must be one of the registered UIDs in the bank (LazyTensor).
        ASSERT_NE(uids.find(source_id), uids.end());
        EXPECT_EQ(source.m_device, device);
        EXPECT_FALSE(source.m_weights.expired());

    }
    {
        auto handle = bank_instance.get_handle();
        // The handle's context must match the shared context used to create the bank instance.
        // REQUIREMENT: shared_context can be available for further use for serialization/deserialization
        ASSERT_EQ(handle.context.get(), shared_context.get());
    }
    auto statAfterFirstEvaluate = this->m_remote_context->get_statistics_snapshot();
     if (device != "CPU") {
        // The number of imported remote tensors must increase by the number of registered UIDs after the first evaluation.
        ASSERT_EQ(statAfterFirstEvaluate.imported_remote_tensors - statBeforeEvaluate.imported_remote_tensors, uids.size());
    }

    // Add one more "shared transformation" (aka constant) to the bank
    constants.push_back(std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                         ov::Shape{2},
                                                         std::vector<float>{5.0f, 6.0f}));
    for (const auto& constant : constants) {
        uids.insert(bank_instance.registerLT(LazyTensor(constant), device));
    }
    bank_instance.evaluate_and_allocate();

    // REQUIREMENT: Double registration of the same "shared transformation" in ov::weight_sharing::Context
    // is forbidden as well as the double registration in NPUW Bank.
    ASSERT_EQ(shared_context->m_runtime_sources.size(), uids.size());
    const auto sources_after_second_evaluate = shared_context->m_runtime_sources;
    for (const auto& [source_id, source] : sources_after_second_evaluate) {
        // REQUIREMENT reinforcement: "source_id" from ov::weight_sharing::Context must match UID of "shared transformation"
        ASSERT_NE(uids.find(source_id), uids.end());
        EXPECT_EQ(source.m_device, device);
        EXPECT_FALSE(source.m_weights.expired());
    }
    {
        auto handle = bank_instance.get_handle();
        // The handle's context must stay the same
        // REQUIREMENT: shared_context can be available for further use for serialization/deserialization
        ASSERT_EQ(handle.context.get(), shared_context.get());
    }
    auto statAfterSecondEvaluate = this->m_remote_context->get_statistics_snapshot();
    if (device != "CPU") {
        // The number of imported remote tensors must increase by the number of registered UIDs after the second evaluation.
        ASSERT_EQ(statAfterSecondEvaluate.imported_remote_tensors - statBeforeEvaluate.imported_remote_tensors, uids.size());
    }
}

TEST_P(WeightsBankTest, EvaluationFallsBackWithoutWeightSharingContext) {
    using ov::npuw::weights::Bank;
    using ov::npuw::weights::LazyTensor;

    const auto& device = GetParam();
    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                           ov::Shape{2},
                                                           std::vector<float>{5.0f, 6.0f});
    Bank bank(m_core, device, "bank-without-context");
    const auto uid = bank.registerLT(LazyTensor(constant), device);

    EXPECT_NO_THROW(bank.evaluate_and_allocate());
    const auto values = read_f32(bank.get(uid, device));
    EXPECT_EQ(values[0], 5.0f);
    EXPECT_EQ(values[1], 6.0f);
}

INSTANTIATE_TEST_SUITE_P(Devices,
                         WeightsBankTest,
                         ::testing::Values("CPU", "NPU"),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });