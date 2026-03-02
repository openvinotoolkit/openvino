// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace ov;

TEST(DisablePrecisionConversionAPITest, no_attribute_returns_false) {
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 10});
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1});
    auto node = std::make_shared<op::v0::MatMul>(data_1, data_2);

    // Node with no DisablePrecisionConversion at all
    ASSERT_FALSE(is_compression_disabled_to(node, element::f16));
    ASSERT_FALSE(is_compression_disabled_from_to(node, element::f32, element::f16));
    ASSERT_FALSE(is_compression_disabled_from_to(node, element::dynamic, element::f16));
}

TEST(DisablePrecisionConversionAPITest, specific_from_to_without_dynamic_entry) {
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 10});
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1});
    auto node = std::make_shared<op::v0::MatMul>(data_1, data_2);

    // Only f32->f16, no dynamic entry
    disable_compression_from_to(node, element::f32, element::f16);

    ASSERT_FALSE(is_compression_disabled_to(node, element::f16));
    ASSERT_TRUE(is_compression_disabled_from_to(node, element::f32, element::f16));
    ASSERT_FALSE(is_compression_disabled_from_to(node, element::f32, element::bf16));
    ASSERT_FALSE(is_compression_disabled_from_to(node, element::f64, element::f16));
}

TEST(DisablePrecisionConversionAPITest, enable_erases_dynamic_entry) {
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 10});
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1});
    auto node = std::make_shared<op::v0::MatMul>(data_1, data_2);

    // Add dynamic->f16, then remove it. dynamic entry should be erased
    disable_compression_to(node, element::f16);

    ASSERT_TRUE(is_compression_disabled_to(node, element::f16));
    ASSERT_TRUE(node->get_rt_info().count(DisablePrecisionConversion::get_type_info_static()));

    const auto& attr_before =
        node->get_rt_info().at(DisablePrecisionConversion::get_type_info_static()).as<DisablePrecisionConversion>();
    ASSERT_EQ(attr_before.m_disabled_precisions.count(element::dynamic), 1);

    enable_compression_to(node, element::f16);

    const auto& attr_after =
        node->get_rt_info().at(DisablePrecisionConversion::get_type_info_static()).as<DisablePrecisionConversion>();
    ASSERT_EQ(attr_after.m_disabled_precisions.count(element::dynamic), 0);

    ASSERT_FALSE(is_compression_disabled_to(node, element::f16));
}

TEST(DisablePrecisionConversionAPITest, specific_from_dynamic_to_blocks_all_targets) {
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 10});
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1});
    auto node = std::make_shared<op::v0::MatMul>(data_1, data_2);

    // f32->dynamic means block f32 to anything
    disable_compression_from_to(node, element::f32, element::dynamic);

    ASSERT_TRUE(is_compression_disabled_from_to(node, element::f32, element::f16));
    ASSERT_TRUE(is_compression_disabled_from_to(node, element::f32, element::bf16));
    ASSERT_TRUE(is_compression_disabled_from_to(node, element::f32, element::i8));

    // Other source types are not blocked
    ASSERT_FALSE(is_compression_disabled_from_to(node, element::f64, element::f16));
    ASSERT_FALSE(is_compression_disabled_to(node, element::f16));
}

TEST(DisablePrecisionConversionAPITest, dynamic_to_dynamic_blocks_everything) {
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 10});
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1});
    auto node = std::make_shared<op::v0::MatMul>(data_1, data_2);

    // dynamic->dynamic means block all conversions
    disable_compression_from_to(node, element::dynamic, element::dynamic);

    ASSERT_TRUE(is_compression_disabled_to(node, element::f16));
    ASSERT_TRUE(is_compression_disabled_to(node, element::bf16));
    ASSERT_TRUE(is_compression_disabled_from_to(node, element::f32, element::f16));
    ASSERT_TRUE(is_compression_disabled_from_to(node, element::f64, element::bf16));
    ASSERT_TRUE(is_compression_disabled_from_to(node, element::f32, element::i8));
}
