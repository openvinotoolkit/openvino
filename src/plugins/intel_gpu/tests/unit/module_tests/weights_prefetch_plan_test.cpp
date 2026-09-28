// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "intel_gpu/plugin/weights_prefetch_plan.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::intel_gpu::tests {

namespace {

class MockMappedMemory : public ov::MappedMemory {
public:
    explicit MockMappedMemory(size_t size) : m_data(size), m_id(1) {}

    char* data() noexcept override {
        return m_data.data();
    }

    size_t size() const noexcept override {
        return m_data.size();
    }

    uint64_t get_id() const noexcept override {
        return m_id;
    }

    MOCK_METHOD(void, hint_evict, (size_t offset, size_t size), (noexcept, override));
    MOCK_METHOD(void, hint_prefetch, (size_t offset, size_t size), (override));
    MOCK_METHOD(void, hint_prefetch_async, (size_t offset, size_t size), (override));
    MOCK_METHOD(void, wait_prefetch, (size_t offset, size_t size), (noexcept, override));

private:
    std::vector<char> m_data;
    uint64_t m_id;
};

std::shared_ptr<ov::op::v0::Constant> make_constant(const std::shared_ptr<MockMappedMemory>& mapping,
                                                    size_t offset,
                                                    size_t size) {
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapping->data() + offset,
                                                                                        size,
                                                                                        mapping);
    return std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                   ov::Shape{size / sizeof(float)},
                                                   std::static_pointer_cast<ov::AlignedBuffer>(buffer));
}

}  // namespace

TEST(weights_prefetch_plan, offload_gate_disables_configured_prefetch) {
    EXPECT_EQ(get_mmap_weights_prefetch_budget(false), 0u);
}

TEST(weights_prefetch_plan, zero_budget_does_not_touch_mmap) {
    auto mapping = std::make_shared<MockMappedMemory>(2048);
    auto lhs = make_constant(mapping, 0, 1024);
    auto rhs = make_constant(mapping, 1024, 1024);
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    EXPECT_CALL(*mapping, hint_prefetch_async(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mapping, wait_prefetch(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mapping, hint_evict(testing::_, testing::_)).Times(0);

    ConstantFoldingPrefetchPlan plan({lhs, rhs, add}, 0);
    plan.prime();
    plan.advance(add);
    plan.finish();
}

TEST(weights_prefetch_plan, constant_folding_disabled_consumer_is_not_planned) {
    auto mapping = std::make_shared<MockMappedMemory>(2048);
    auto lhs = make_constant(mapping, 0, 1024);
    auto rhs = make_constant(mapping, 1024, 1024);
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    ov::pass::disable_constant_folding(add);

    EXPECT_CALL(*mapping, hint_prefetch_async(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mapping, wait_prefetch(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*mapping, hint_evict(testing::_, testing::_)).Times(0);

    ConstantFoldingPrefetchPlan plan({lhs, rhs, add}, 2048);
    plan.prime();
    plan.advance(add);
    plan.finish();
}

TEST(weights_prefetch_plan, program_builder_plan_evicts_region_after_consumption) {
    auto mapping = std::make_shared<MockMappedMemory>(1024);
    auto constant = make_constant(mapping, 0, 1024);

    testing::InSequence sequence;
    EXPECT_CALL(*mapping, hint_prefetch_async(0, 1024));
    EXPECT_CALL(*mapping, wait_prefetch(0, 1024));
    EXPECT_CALL(*mapping, hint_evict(0, 1024));

    WeightsPrefetchPlan plan({constant}, 1024);
    plan.prime();
    plan.before(constant);
    plan.after(constant);
}

TEST(weights_prefetch_plan, program_builder_plan_cleans_up_outstanding_region) {
    auto mapping = std::make_shared<MockMappedMemory>(1024);
    auto constant = make_constant(mapping, 0, 1024);

    testing::InSequence sequence;
    EXPECT_CALL(*mapping, hint_prefetch_async(0, 1024));
    EXPECT_CALL(*mapping, wait_prefetch(0, 1024));
    EXPECT_CALL(*mapping, hint_evict(0, 1024));

    {
        WeightsPrefetchPlan plan({constant}, 1024);
        plan.prime();
    }
}

TEST(weights_prefetch_plan, all_inputs_of_one_consumer_make_progress_when_they_exceed_budget) {
    auto mapping = std::make_shared<MockMappedMemory>(2048);
    auto lhs = make_constant(mapping, 0, 1024);
    auto rhs = make_constant(mapping, 1024, 1024);
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    testing::InSequence sequence;
    EXPECT_CALL(*mapping, hint_prefetch_async(0, 1024));
    EXPECT_CALL(*mapping, wait_prefetch(0, 1024));
    EXPECT_CALL(*mapping, hint_prefetch_async(1024, 1024));
    EXPECT_CALL(*mapping, wait_prefetch(1024, 1024));
    EXPECT_CALL(*mapping, hint_evict(0, 1024));
    EXPECT_CALL(*mapping, hint_evict(1024, 1024));

    ConstantFoldingPrefetchPlan plan({lhs, rhs, add}, 1024);
    plan.prime();
    plan.advance(add);
    plan.finish();
}

TEST(weights_prefetch_plan, overlapping_inputs_do_not_start_concurrent_prefetch) {
    auto mapping = std::make_shared<MockMappedMemory>(1024);
    auto lhs = make_constant(mapping, 0, 1024);
    auto rhs = make_constant(mapping, 0, 1024);
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);

    EXPECT_CALL(*mapping, hint_prefetch_async(0, 1024)).Times(1);
    EXPECT_CALL(*mapping, wait_prefetch(0, 1024)).Times(2);
    EXPECT_CALL(*mapping, hint_evict(0, 1024)).Times(2);

    ConstantFoldingPrefetchPlan plan({lhs, rhs, add}, 2048);
    plan.prime();
    plan.advance(add);
    plan.finish();
}

TEST(weights_prefetch_plan, shared_region_is_evicted_only_after_its_last_consumer) {
    auto mapping = std::make_shared<MockMappedMemory>(1024);
    auto shared = make_constant(mapping, 0, 1024);
    auto one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256}, {1.0f});
    auto two = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256}, {2.0f});
    auto add = std::make_shared<ov::op::v1::Add>(shared, one);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(shared, two);

    testing::InSequence sequence;
    EXPECT_CALL(*mapping, hint_prefetch_async(0, 1024));
    EXPECT_CALL(*mapping, wait_prefetch(0, 1024));
    EXPECT_CALL(*mapping, hint_evict(0, 1024));

    ConstantFoldingPrefetchPlan plan({shared, one, add, two, multiply}, 1024);
    plan.prime();
    plan.advance(add);
    plan.advance(multiply);
    plan.finish();
}

TEST(weights_prefetch_plan, nested_constant_folding_uses_an_independent_body_plan) {
    auto mapping = std::make_shared<MockMappedMemory>(1024);
    auto mapped_data = make_constant(mapping, 0, 1024);
    auto one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256}, {1.0f});
    auto add = std::make_shared<ov::op::v1::Add>(mapped_data, one);
    auto indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, 1}, {0});
    auto gather = std::make_shared<ov::op::v8::GatherND>(add, indices, 0);
    ASSERT_FALSE(gather->has_evaluate());

    auto iteration = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});
    auto body_condition = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{});
    auto body_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto condition_result = std::make_shared<ov::op::v0::Result>(body_condition);
    auto value_result = std::make_shared<ov::op::v0::Result>(gather);
    auto body = std::make_shared<ov::Model>(ov::ResultVector{condition_result, value_result},
                                            ov::ParameterVector{iteration, body_condition, body_input});

    auto trip_count = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto initial_condition = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {true});
    auto outer_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, initial_condition);
    loop->set_function(body);
    loop->set_special_body_ports({0, 0});
    loop->set_merged_input(body_condition, initial_condition, condition_result);
    loop->set_invariant_input(body_input, outer_input);
    auto scan = loop->get_concatenated_slices(value_result, 0, 1, 1, -1, 0);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{scan}, ov::ParameterVector{outer_input});

    testing::InSequence sequence;
    EXPECT_CALL(*mapping, hint_prefetch_async(0, 1024));
    EXPECT_CALL(*mapping, wait_prefetch(0, 1024));
    EXPECT_CALL(*mapping, hint_evict(0, 1024));

    auto observer = std::make_shared<ConstantFoldingPrefetchState>(1024);
    ov::pass::Manager manager;
    manager.get_pass_config()->set_pass_extension<ov::pass::ConstantFolding,
                                                   ov::pass::ConstantFolding::Observer>(observer);
    manager.register_pass<ov::pass::ConstantFolding>();
    EXPECT_NO_THROW(manager.run_passes(model));
    observer->finish();
}

}  // namespace ov::intel_gpu::tests
