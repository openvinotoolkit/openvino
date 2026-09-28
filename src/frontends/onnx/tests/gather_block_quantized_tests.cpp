// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "transformations/op_conversions/convert_gather_to_compressed.hpp"

using namespace ov::frontend::onnx::tests;

// These tests verify that the com.microsoft.GatherBlockQuantized translator emits a decomposition that the
// plugins' ConvertGatherToGatherCompressed pass actually recognizes and folds into the internal
// GatherCompressed op. Converting the real .onnx model (rather than a hand-built graph) exercises the exact
// pattern the frontend produces.
namespace {
std::shared_ptr<ov::Model> convert_and_fuse(const std::string& model_path) {
    auto model = convert_model(model_path);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertGatherToGatherCompressed>();
    manager.run_passes(model);
    return model;
}
}  // namespace

// int4 data, no zero_points -> symmetric decomposition (Multiply(Convert(data), scale) -> Reshape -> Gather).
// The whole subgraph must fold into a single GatherCompressed with no leftover Gather/Multiply.
TEST(ONNXFeGatherBlockQuantized, int4_no_zp_fuses_into_gather_compressed) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_and_fuse("com.microsoft/gather_block_quantized_int4.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::op::internal::GatherCompressed>(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Gather>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Multiply>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Subtract>(model), 0);
}

// int4 data with zero_points -> asymmetric decomposition (Multiply(Subtract(Convert(data), zp), scale) ->
// Reshape -> Gather). It must fold into a GatherCompressed (with a zero_point input) and leave no standalone
// Gather/Multiply/Subtract behind.
TEST(ONNXFeGatherBlockQuantized, int4_with_zp_fuses_into_gather_compressed) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_and_fuse("com.microsoft/gather_block_quantized_int4_zp.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::op::internal::GatherCompressed>(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Gather>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Multiply>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Subtract>(model), 0);
}

// uint4 data, no zero_points -> same symmetric fusable pattern as int4.
TEST(ONNXFeGatherBlockQuantized, uint4_no_zp_fuses_into_gather_compressed) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_and_fuse("com.microsoft/gather_block_quantized_uint4.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::op::internal::GatherCompressed>(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Gather>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Multiply>(model), 0);
}

// float16 scales -> the dequant runs in f16 and the Gather is on f16 directly (no trailing Convert to f32);
// the pattern must still fold into GatherCompressed.
TEST(ONNXFeGatherBlockQuantized, int4_f16_scales_fuses_into_gather_compressed) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_and_fuse("com.microsoft/gather_block_quantized_int4_f16.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::op::internal::GatherCompressed>(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::op::v8::Gather>(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Multiply>(model), 0);
}
