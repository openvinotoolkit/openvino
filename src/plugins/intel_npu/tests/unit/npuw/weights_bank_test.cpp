// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "lazy_tensor.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "weights_bank.hpp"

TEST(WeightsBankTest, CpuEvaluationReusesRuntimeWeightSources) {
    using ov::npuw::weights::Bank;
    using ov::npuw::weights::LazyTensor;

    auto first = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                        ov::Shape{2},
                                                        std::vector<float>{1.0f, 2.0f});
    auto second = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                         ov::Shape{2},
                                                         std::vector<float>{3.0f, 4.0f});
    auto shared_context = std::make_shared<ov::weight_sharing::Context>();

    Bank first_bank(nullptr, "CPU", "first-bank", shared_context);
    const auto first_uid = first_bank.registerLT(LazyTensor(first), "CPU");
    const auto second_uid = first_bank.registerLT(LazyTensor(second), "CPU");
    first_bank.evaluate_and_allocate();

    ASSERT_EQ(shared_context->m_runtime_sources.size(), 2);
    const auto sources_before = shared_context->m_runtime_sources;
    for (const auto& [source_id, source] : sources_before) {
        EXPECT_GT(source_id, ov::weight_sharing::invalid_source_id);
        EXPECT_EQ(source.m_device, "CPU");
        EXPECT_FALSE(source.m_weights.expired());
    }
    const auto* first_tensor_data = first_bank.get(second_uid, "CPU").data();

    first_bank.evaluate_and_allocate();
    ASSERT_EQ(shared_context->m_runtime_sources.size(), sources_before.size());

    Bank second_bank(nullptr, "CPU", "second-bank", shared_context);
    EXPECT_EQ(second_bank.registerLT(LazyTensor(first), "CPU"), first_uid);
    EXPECT_EQ(second_bank.registerLT(LazyTensor(second), "CPU"), second_uid);
    second_bank.evaluate_and_allocate();

    ASSERT_EQ(shared_context->m_runtime_sources.size(), sources_before.size());
    for (const auto& [source_id, source_before] : sources_before) {
        const auto source_after = ov::weight_sharing::get_runtime_weight_source(*shared_context, source_id);
        EXPECT_EQ(source_after.m_device, source_before.m_device);
        EXPECT_EQ(source_after.m_weights.lock(), source_before.m_weights.lock());
    }

    auto second_buffer = ov::weight_sharing::get_runtime_weight_source(*shared_context, second_uid).m_weights.lock();
    ASSERT_TRUE(second_buffer);
    EXPECT_EQ(first_tensor_data, second_buffer->get_ptr());
    EXPECT_EQ(second_bank.get(second_uid, "CPU").data(), second_buffer->get_ptr());
    EXPECT_EQ(first_bank.get(second_uid, "CPU").data<float>()[0], 3.0f);
    EXPECT_EQ(second_bank.get(second_uid, "CPU").data<float>()[1], 4.0f);
}

TEST(WeightsBankTest, CpuEvaluationFallsBackWithoutWeightSharingContext) {
    using ov::npuw::weights::Bank;
    using ov::npuw::weights::LazyTensor;

    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                           ov::Shape{2},
                                                           std::vector<float>{5.0f, 6.0f});
    Bank bank(nullptr, "CPU", "bank-without-context");
    const auto uid = bank.registerLT(LazyTensor(constant), "CPU");

    EXPECT_NO_THROW(bank.evaluate_and_allocate());
    const auto tensor = bank.get(uid, "CPU");
    EXPECT_EQ(tensor.data<float>()[0], 5.0f);
    EXPECT_EQ(tensor.data<float>()[1], 6.0f);
}