// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_grouped_depth_to_space.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "intel_gpu/op/grouped_depth_to_space.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"

namespace {

std::shared_ptr<ov::op::v0::Constant> i64(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

std::shared_ptr<ov::op::v0::Constant> i64_scalar(int64_t value) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

std::shared_ptr<ov::Model> make_dup_up_model(bool with_crop, bool valid_transpose_order = true, bool shape_derived_range_stop = false) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 2, 3, 5});

    ov::Output<ov::Node> range_stop = i64_scalar(4);
    if (shape_derived_range_stop) {
        auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
        range_stop = std::make_shared<ov::op::v8::Gather>(input_shape, i64_scalar(1), i64_scalar(0));
    }
    auto range = std::make_shared<ov::op::v4::Range>(i64_scalar(0), range_stop, i64_scalar(1), ov::element::i64);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(range, i64({0}));
    auto tile = std::make_shared<ov::op::v0::Tile>(unsqueeze, i64({4, 1}));
    auto indices_transpose = std::make_shared<ov::op::v1::Transpose>(tile, i64({1, 0}));
    auto indices = std::make_shared<ov::op::v1::Reshape>(indices_transpose, i64({-1}), false);
    auto gather = std::make_shared<ov::op::v8::Gather>(input, indices, i64({1}));

    auto factor_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{i64({1}), i64({2}), i64({2}), i64({2}), i64({2}), i64({2}), i64({3, 5})}, 0);
    auto factor_reshape = std::make_shared<ov::op::v1::Reshape>(gather, factor_shape, false);
    const std::vector<int64_t> transpose_order =
        valid_transpose_order ? std::vector<int64_t>{0, 1, 5, 2, 6, 3, 7, 4} : std::vector<int64_t>{0, 1, 5, 2, 6, 4, 7, 3};
    auto transpose = std::make_shared<ov::op::v1::Transpose>(factor_reshape, i64(transpose_order));

    auto output_height = std::make_shared<ov::op::v1::Multiply>(i64({3}), i64({2}));
    auto output_width = std::make_shared<ov::op::v1::Multiply>(i64({5}), i64({2}));
    auto output_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{i64({1}), i64({2}), i64({4}), output_height, output_width}, 0);
    auto output_reshape = std::make_shared<ov::op::v1::Reshape>(transpose, output_shape, false);

    ov::Output<ov::Node> output = output_reshape;
    if (with_crop) {
        output = std::make_shared<ov::op::v8::Slice>(output_reshape, i64({1}), i64({std::numeric_limits<int64_t>::max()}), i64({1}), i64({2}));
    }
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
}

std::shared_ptr<ov::intel_gpu::op::GroupedDepthToSpace> run_fusion(const std::shared_ptr<ov::Model>& model, bool fold_constants = false) {
    ov::pass::Manager manager;
    if (fold_constants) {
        manager.register_pass<ov::pass::ConstantFolding>();
    }
    manager.register_pass<ov::intel_gpu::FuseGroupedDepthToSpace>();
    manager.run_passes(model);
    return ov::as_type_ptr<ov::intel_gpu::op::GroupedDepthToSpace>(model->get_results().front()->get_input_node_shared_ptr(0));
}

TEST(FuseGroupedDepthToSpaceTest, FusesWithoutTemporalCrop) {
    const auto grouped = run_fusion(make_dup_up_model(false));

    ASSERT_NE(grouped, nullptr);
    EXPECT_EQ(grouped->get_factor_t(), 2u);
    EXPECT_EQ(grouped->get_factor_s(), 2u);
    EXPECT_EQ(grouped->get_output_channels(), 2u);
    EXPECT_EQ(grouped->get_crop_begin_t(), 0u);
    EXPECT_EQ(grouped->get_output_shape(0), (ov::Shape{1, 2, 4, 6, 10}));
}

TEST(FuseGroupedDepthToSpaceTest, FusesTemporalCrop) {
    const auto grouped = run_fusion(make_dup_up_model(true));

    ASSERT_NE(grouped, nullptr);
    EXPECT_EQ(grouped->get_crop_begin_t(), 1u);
    EXPECT_EQ(grouped->get_output_shape(0), (ov::Shape{1, 2, 3, 6, 10}));
}

TEST(FuseGroupedDepthToSpaceTest, FusesShapeDerivedRangeStop) {
    EXPECT_NE(run_fusion(make_dup_up_model(false, true, true)), nullptr);
}

TEST(FuseGroupedDepthToSpaceTest, FusesConstantFoldedInputs) {
    EXPECT_NE(run_fusion(make_dup_up_model(false, true, true), true), nullptr);
}

TEST(FuseGroupedDepthToSpaceTest, RejectsDifferentTransposeOrder) {
    EXPECT_EQ(run_fusion(make_dup_up_model(false, false)), nullptr);
}

}  // namespace
