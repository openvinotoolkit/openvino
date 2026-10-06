// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/util/mmap_object.hpp"
#include "shared_weights_assigner.hpp"
#include "shared_weights_context_extractor.hpp"
#include "shared_weights_producer.hpp"

namespace {

using SharedConstant = ov::intel_npu::transformations::SharedWeightsAssigner::SharedConstant;
using PartitionedConstants = ov::intel_npu::transformations::SharedWeightsAssigner::PartitionedConstants;
using SharedSourcesWithConstants = ov::intel_npu::transformations::SharedWeightsAssigner::SharedSourcesWithConstants;

std::shared_ptr<ov::Model> make_test_model(
    size_t element_count,
    std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>>& by_name) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{element_count});

    std::vector<uint8_t> data_a(element_count, 1);
    std::vector<uint8_t> data_b(element_count, 2);
    std::vector<uint8_t> data_c(element_count, 3);

    auto c1 = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{element_count}, data_a);
    auto c2 = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{element_count}, data_b);
    auto c3 = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{element_count}, data_c);

    c1->set_friendly_name("c1");
    c2->set_friendly_name("c2");
    c3->set_friendly_name("c3");

    by_name["c1"] = c1;
    by_name["c2"] = c2;
    by_name["c3"] = c3;

    auto add1 = std::make_shared<ov::op::v1::Add>(input, c1);
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, c2);
    auto mul = std::make_shared<ov::op::v1::Multiply>(add2, c3);

    auto result = std::make_shared<ov::op::v0::Result>(mul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> collect_named_constants(
    const std::shared_ptr<ov::Model>& model) {
    std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> result;
    for (const auto& op : model->get_ops()) {
        auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
        if (!constant) {
            continue;
        }
        result[constant->get_friendly_name()] = constant;
    }
    return result;
}

ov::intel_npu::transformations::SharedWeightsAssigner::Options make_options(size_t max_source_size) {
    ov::intel_npu::transformations::SharedWeightsAssigner::Options options;
    options.single_weight_shared_source_size_max = max_source_size;
    options.preserve_weightless_cache_attr = true;
    options.source_id_generator = [id = static_cast<size_t>(100)]() mutable {
        return id++;
    };
    return options;
}

TEST(SharedWeightsAssignerTest, OptionsDefaultToMaximumSharedSourceSize) {
    const ov::intel_npu::transformations::SharedWeightsAssigner::Options options{};
    EXPECT_EQ(options.single_weight_shared_source_size_max, std::numeric_limits<size_t>::max());
}

TEST(SharedWeightsAssignerTest, CollectAndPartitionProvidesStatsWithoutMutation) {
    const size_t page_size = static_cast<size_t>(ov::util::get_system_page_size());

    std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> original_by_name;
    auto model = make_test_model(page_size, original_by_name);
    const size_t model_constants_count = original_by_name.size();

    size_t expected_constant_partition_count = (model_constants_count - 1);
    ov::intel_npu::transformations::SharedWeightsAssigner assigner(
        make_options(expected_constant_partition_count * page_size));
    auto collect_result = assigner.collect_and_partition(model);

    ASSERT_EQ(collect_result.statistic.collected_constants_count, model_constants_count);
    ASSERT_EQ(collect_result.statistic.partition_constant_counts.size(), expected_constant_partition_count);
    ASSERT_EQ(collect_result.statistic.partition_constant_counts[0], (model_constants_count - 1));
    ASSERT_EQ(collect_result.statistic.partition_constant_counts[1], 1u);
    ASSERT_EQ(collect_result.statistic.constant_cannot_be_shared_count, 0u);
    ASSERT_EQ(collect_result.statistic.total_shared_constant_bytes, model_constants_count * page_size);
    ASSERT_EQ(collect_result.statistic.total_non_shared_constant_bytes_released, model_constants_count * page_size);

    auto current_by_name = collect_named_constants(model);
    ASSERT_EQ(current_by_name.size(), model_constants_count);
    for (auto& [name, constant] : current_by_name) {
        ASSERT_EQ(constant.get(), original_by_name.at(name).get());
    }
}

TEST(SharedWeightsAssignerTest, MutateModelWithConstantSharingReturnsExpectedBuffers) {
    const size_t page_size = static_cast<size_t>(ov::util::get_system_page_size());

    std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> original_by_name;
    auto model = make_test_model(page_size, original_by_name);
    const size_t model_constants_count = original_by_name.size();

    const size_t expected_constant_partition_count = model_constants_count - 1;
    ov::intel_npu::transformations::SharedWeightsAssigner assigner(
        make_options(expected_constant_partition_count * page_size));
    auto collect_result = assigner.collect_and_partition(model);
    auto shared_sources_with_constants =
        assigner.mutate_model_with_constant_sharing(std::move(collect_result.partitioned_constants));

    ASSERT_EQ(shared_sources_with_constants.size(), expected_constant_partition_count);
    EXPECT_EQ(shared_sources_with_constants[0].first->size(), (model_constants_count - 1) * page_size);
    EXPECT_EQ(shared_sources_with_constants[1].first->size(), page_size);
    EXPECT_EQ(shared_sources_with_constants[0].second.size(), model_constants_count - 1);
    EXPECT_EQ(shared_sources_with_constants[1].second.size(), 1u);

    // All returned constants should point inside their partition buffer ranges.
    for (const auto& [source, shared_constants] : shared_sources_with_constants) {
        const auto source_begin = reinterpret_cast<uintptr_t>(source->get_ptr<char>());
        const auto source_end = source_begin + source->size();
        for (const auto& shared_constant : shared_constants) {
            const auto data_begin = reinterpret_cast<uintptr_t>(shared_constant->get_data_ptr());
            const auto data_end = data_begin + shared_constant->get_byte_size();
            EXPECT_GE(data_begin, source_begin);
            EXPECT_LE(data_end, source_end);
        }
    }

    auto mutated_by_name = collect_named_constants(model);
    ASSERT_EQ(mutated_by_name.size(), model_constants_count);
    for (auto& [name, constant] : mutated_by_name) {
        ASSERT_NE(constant.get(), original_by_name.at(name).get());
    }

    auto model_context =
        ov::intel_npu::transformations::SharedWeightsContextExtractor::extract_weight_sharing_context(model);
    ASSERT_EQ(model_context->m_runtime_sources.size(), expected_constant_partition_count);
    EXPECT_TRUE(model_context->m_cache_sources.empty());
    size_t model_context_constant_count = 0;
    for (const auto& registry_entry : model_context->m_weight_registry) {
        model_context_constant_count += registry_entry.second.size();
    }
    EXPECT_EQ(model_context_constant_count, model_constants_count);

    auto shared_sources_context =
        ov::intel_npu::transformations::SharedWeightsContextExtractor::extract_weight_sharing_context(
            shared_sources_with_constants);
    ASSERT_EQ(shared_sources_context->m_runtime_sources.size(), expected_constant_partition_count);
    EXPECT_TRUE(shared_sources_context->m_cache_sources.empty());
    size_t shared_sources_constant_count = 0;
    for (const auto& registry_entry : shared_sources_context->m_weight_registry) {
        shared_sources_constant_count += registry_entry.second.size();
    }
    EXPECT_EQ(shared_sources_constant_count, model_constants_count);
}

TEST(SharedWeightsProducerTest, EmptyPropertyDoesNotMutateModel) {
    std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> original_by_name;
    auto model = make_test_model(static_cast<size_t>(ov::util::get_system_page_size()), original_by_name);

    auto [sources, context] =
        ov::intel_npu::transformations::assign_shared_weight_to_model_if_possible(model, ov::Any{});

    EXPECT_TRUE(sources.empty());
    EXPECT_EQ(context, nullptr);
    auto current_by_name = collect_named_constants(model);
    for (const auto& [name, constant] : original_by_name) {
        EXPECT_EQ(current_by_name.at(name), constant);
    }
}

TEST(SharedWeightsProducerTest, ReturnsSourcesAndContextForSharedConstants) {
    const size_t page_size = static_cast<size_t>(ov::util::get_system_page_size());
    std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> original_by_name;
    auto model = make_test_model(page_size, original_by_name);

    auto [sources, context] =
        ov::intel_npu::transformations::assign_shared_weight_to_model_if_possible(model, std::string("NPU"));

    ASSERT_EQ(sources.size(), 1u);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->m_runtime_sources.size(), sources.size());
    EXPECT_EQ(sources.front()->size(), original_by_name.size() * page_size);
    const auto source_id = sources.front()->get_descriptor()->get_id();
    EXPECT_EQ(context->m_runtime_sources.at(source_id).m_weights.lock(), sources.front());
    ASSERT_EQ(context->m_weight_registry.count(source_id), 1u);
    EXPECT_EQ(context->m_weight_registry.at(source_id).size(), original_by_name.size());

    auto current_by_name = collect_named_constants(model);
    ASSERT_EQ(current_by_name.size(), original_by_name.size());
    for (const auto& [name, original] : original_by_name) {
        const auto& constant = current_by_name.at(name);
        EXPECT_NE(constant, original);
        EXPECT_EQ(ov::weight_sharing::Extension::get_constant_source_buffer(*constant), sources.front());
    }
}

TEST(SharedWeightsProducerTest, RejectsNonStringProperty) {
    std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> original_by_name;
    auto model = make_test_model(static_cast<size_t>(ov::util::get_system_page_size()), original_by_name);

    EXPECT_THROW(ov::intel_npu::transformations::assign_shared_weight_to_model_if_possible(model, 123), ov::Exception);
}

}  // namespace
