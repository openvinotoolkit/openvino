// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/convert_legacy_precision_attribute.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov::testing {

TEST(ConvertLegacyPrecisionAttributeTest, basic_migration) {
    // Create model with legacy attribute on a node
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 10});
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1});
    auto matmul = std::make_shared<op::v0::MatMul>(data_1, data_2);
    OPENVINO_SUPPRESS_DEPRECATED_START
    disable_fp16_compression(matmul);
    OPENVINO_SUPPRESS_DEPRECATED_END
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data_1, data_2});

    // Verify legacy attribute is present before the pass
    ASSERT_TRUE(matmul->get_rt_info().count(DisableFP16Compression::get_type_info_static()));

    pass::Manager m;
    m.register_pass<pass::ConvertLegacyPrecisionAttribute>();
    bool res = m.run_passes(model);

    ASSERT_TRUE(res);

    const auto& rt_info = matmul->get_rt_info();

    ASSERT_FALSE(rt_info.count(DisableFP16Compression::get_type_info_static()));
    ASSERT_TRUE(rt_info.count(DisablePrecisionConversion::get_type_info_static()));

    // Verify the map content: {dynamic -> {f16}}
    const auto& attr = rt_info.at(DisablePrecisionConversion::get_type_info_static()).as<DisablePrecisionConversion>();
    DisabledPrecisionMap expected = {{element::dynamic, {element::f16}}};
    ASSERT_EQ(attr.m_disabled_precisions, expected);

    ASSERT_TRUE(is_compression_disabled_to(matmul, element::f16));
}

TEST(ConvertLegacyPrecisionAttributeTest, no_legacy_attribute) {
    // Node without any attribute — pass should not modify it
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 10});
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1});
    auto matmul = std::make_shared<op::v0::MatMul>(data_1, data_2);
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data_1, data_2});

    pass::Manager m;
    m.register_pass<pass::ConvertLegacyPrecisionAttribute>();
    bool res = m.run_passes(model);

    ASSERT_FALSE(res);

    const auto& rt_info = matmul->get_rt_info();
    ASSERT_FALSE(rt_info.count(DisableFP16Compression::get_type_info_static()));
    ASSERT_FALSE(rt_info.count(DisablePrecisionConversion::get_type_info_static()));
    ASSERT_FALSE(is_compression_disabled_to(matmul, element::f16));
}

TEST(ConvertLegacyPrecisionAttributeTest, multiple_nodes) {
    // Multiple nodes with legacy attribute
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 10});
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1});
    auto add = std::make_shared<op::v1::Add>(data_1, data_1);
    auto matmul = std::make_shared<op::v0::MatMul>(add, data_2);
    OPENVINO_SUPPRESS_DEPRECATED_START
    disable_fp16_compression(add);
    disable_fp16_compression(matmul);
    OPENVINO_SUPPRESS_DEPRECATED_END
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data_1, data_2});

    pass::Manager m;
    m.register_pass<pass::ConvertLegacyPrecisionAttribute>();
    bool res = m.run_passes(model);

    ASSERT_TRUE(res);

    // Both nodes should be migrated
    ASSERT_FALSE(add->get_rt_info().count(DisableFP16Compression::get_type_info_static()));
    ASSERT_TRUE(add->get_rt_info().count(DisablePrecisionConversion::get_type_info_static()));
    ASSERT_TRUE(is_compression_disabled_to(add, element::f16));

    ASSERT_FALSE(matmul->get_rt_info().count(DisableFP16Compression::get_type_info_static()));
    ASSERT_TRUE(matmul->get_rt_info().count(DisablePrecisionConversion::get_type_info_static()));
    ASSERT_TRUE(is_compression_disabled_to(matmul, element::f16));
}

}  // namespace ov::testing