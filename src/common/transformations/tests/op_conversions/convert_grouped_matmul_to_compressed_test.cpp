// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_compressed.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace {

// Regression test for the u4 -> u8 conversion regression on GroupedMatMul.
// The ConvertGroupedMatMulToGroupedMatMulCompressed transformation itself must
// mark the origin u4 Constants with `keep_const_precision`.
TEST(ConvertGroupedMatMulToGroupedMatMulCompressed, KeepU4WeightsAndZpPrecision_2Dx3D) {
    const ov::Shape wei_shape{4, 256, 128};
    const ov::Shape scale_zp_shape{4, 256, 1};

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 128});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{4});

    auto weights_u4 = ov::op::v0::Constant::create(ov::element::u4, wei_shape, {1});
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_u4, ov::element::f16);

    auto zp_u4 = ov::op::v0::Constant::create(ov::element::u4, scale_zp_shape, {1});
    auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_u4, ov::element::f16);
    auto sub = std::make_shared<ov::op::v1::Subtract>(weights_convert, zp_convert);

    auto scale_const = ov::op::v0::Constant::create(ov::element::f16, scale_zp_shape, {1});
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);

    auto gmm = std::make_shared<ov::op::v17::GroupedMatMul>(data, mul, offsets);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{data, offsets});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertGroupedMatMulToGroupedMatMulCompressed>(
        std::vector<ov::element::Type>{ov::element::u4, ov::element::i4, ov::element::u8, ov::element::i8});
    manager.run_passes(model);

    std::shared_ptr<ov::op::internal::GroupedMatMulCompressed> compressed;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::internal::GroupedMatMulCompressed>(op)) {
            compressed = c;
            break;
        }
    }
    ASSERT_TRUE(compressed) << "Transformation did not produce GroupedMatMulCompressed";

    // Weights (input 1) must remain a u4 Constant and carry keep_const_precision so that
    // ConvertPrecision does not upcast it later in the pipeline.
    auto weights_node = compressed->get_input_node_shared_ptr(1);
    ASSERT_TRUE(ov::is_type<ov::op::v0::Constant>(weights_node))
        << "Expected weights input to be a Constant, got " << weights_node->get_type_name();
    EXPECT_EQ(weights_node->get_output_element_type(0), ov::element::u4);
    EXPECT_TRUE(ov::is_keep_const_precision(weights_node))
        << "u4 weights Constant is not marked with keep_const_precision";

    // Zero-point (input 4 in the 2D x 3D form) is wrapped in a Convert(u4 -> u8) by the
    // transformation (`convert_u4zp_to_u8=true`). The underlying u4 leaf Constant must
    // still be marked so ConvertPrecision does not fold the Convert away by upcasting.
    auto zp_input = compressed->get_input_node_shared_ptr(4);
    std::shared_ptr<ov::Node> zp_leaf = zp_input;
    if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(zp_input)) {
        zp_leaf = convert->get_input_node_shared_ptr(0);
    }
    ASSERT_TRUE(ov::is_type<ov::op::v0::Constant>(zp_leaf))
        << "Expected ZP leaf to be a Constant, got " << zp_leaf->get_type_name();
    EXPECT_EQ(zp_leaf->get_output_element_type(0), ov::element::u4);
    EXPECT_TRUE(ov::is_keep_const_precision(zp_leaf))
        << "u4 zero-point Constant is not marked with keep_const_precision";
}

TEST(ConvertGroupedMatMulToGroupedMatMulCompressed, KeepU4WeightsPrecision_3Dx3D_NoZp) {
    const ov::Shape wei_shape{4, 256, 128};
    const ov::Shape scale_shape{4, 256, 1};

    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{4, -1, 128});

    auto weights_u4 = ov::op::v0::Constant::create(ov::element::u4, wei_shape, {1});
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_u4, ov::element::f16);

    auto scale_const = ov::op::v0::Constant::create(ov::element::f16, scale_shape, {1});
    auto mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);

    auto gmm = std::make_shared<ov::op::v17::GroupedMatMul>(data, mul);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{data});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertGroupedMatMulToGroupedMatMulCompressed>(
        std::vector<ov::element::Type>{ov::element::u4, ov::element::i4, ov::element::u8, ov::element::i8});
    manager.run_passes(model);

    std::shared_ptr<ov::op::internal::GroupedMatMulCompressed> compressed;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::internal::GroupedMatMulCompressed>(op)) {
            compressed = c;
            break;
        }
    }
    ASSERT_TRUE(compressed) << "Transformation did not produce GroupedMatMulCompressed";

    auto weights_node = compressed->get_input_node_shared_ptr(1);
    ASSERT_TRUE(ov::is_type<ov::op::v0::Constant>(weights_node))
        << "Expected weights input to be a Constant, got " << weights_node->get_type_name();
    EXPECT_EQ(weights_node->get_output_element_type(0), ov::element::u4);
    EXPECT_TRUE(ov::is_keep_const_precision(weights_node))
        << "u4 weights Constant is not marked with keep_const_precision";
}

}  // namespace
