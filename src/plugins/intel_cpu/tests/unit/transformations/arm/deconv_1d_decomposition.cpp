// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/deconv_1d_decomposition.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov::intel_cpu;

static std::shared_ptr<ov::Model> createDynamicDeconv3D(const ov::PartialShape& input_shape,
                                                        const ov::Shape& weights_shape,
                                                        size_t stride = 1,
                                                        int64_t pad_begin = 0,
                                                        int64_t pad_end = 0) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
    auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});

    auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                        weights,
                                                                        ov::Strides{stride},
                                                                        ov::CoordinateDiff{static_cast<ptrdiff_t>(pad_begin)},
                                                                        ov::CoordinateDiff{static_cast<ptrdiff_t>(pad_end)},
                                                                        ov::Strides{1});

    return std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input});
}

class Deconv1DDecompositionTest : public TransformationTestsF {
protected:
    static std::shared_ptr<ov::Model> createDeconv1D(const ov::Shape& input_shape,
                                                     const ov::Shape& weights_shape,
                                                     size_t stride = 1,
                                                     int64_t pad_begin = 0,
                                                     int64_t pad_end = 0,
                                                     size_t dilation = 1,
                                                     ov::op::PadType pad_type = ov::op::PadType::EXPLICIT,
                                                     int64_t output_padding = 0,
                                                     ov::element::Type element_type = ov::element::f32) {
        auto input = std::make_shared<ov::opset1::Parameter>(element_type, input_shape);
        auto weights = ov::opset1::Constant::create(element_type, weights_shape, {1});

        auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                            weights,
                                                                            ov::Strides{stride},
                                                                            ov::CoordinateDiff{static_cast<ptrdiff_t>(pad_begin)},
                                                                            ov::CoordinateDiff{static_cast<ptrdiff_t>(pad_end)},
                                                                            ov::Strides{dilation},
                                                                            pad_type,
                                                                            ov::CoordinateDiff{static_cast<ptrdiff_t>(output_padding)});

        return std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input});
    }

    static std::shared_ptr<ov::Model> createDeconv1DReference(const ov::Shape& input_shape,
                                                               const ov::Shape& weights_shape,
                                                               size_t stride = 1,
                                                               int64_t pad_begin = 0,
                                                               int64_t pad_end = 0,
                                                               size_t dilation = 1,
                                                               ov::op::PadType pad_type = ov::op::PadType::EXPLICIT,
                                                               int64_t output_padding = 0,
                                                               ov::element::Type element_type = ov::element::f32) {
        auto input = std::make_shared<ov::opset1::Parameter>(element_type, input_shape);
        auto weights = ov::opset1::Constant::create(element_type, weights_shape, {1});

        // Build expected decomposed graph manually
        // The transformation decomposes 1D deconv into: Reshape -> Upsampling -> Padding -> Conv2D -> Reshape

        // Step 1: Reshape input from 3D to 4D [N, C, L] -> [N, C, L, 1]
        auto input_4d_shape = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 128, 100, 1});
        auto input_4d = std::make_shared<ov::opset1::Reshape>(input, input_4d_shape, false);

        // Step 2: For stride=1, no upsampling needed, so result = input_4d
        ov::Output<ov::Node> result = input_4d;

        // Step 3: Padding - for kernel size 5, pad_left = 5-1 = 4, pad_right = 5-1 = 4
        auto pads_begin = ov::opset1::Constant::create(ov::element::i32, {4}, {0, 0, 4, 0});
        auto pads_end = ov::opset1::Constant::create(ov::element::i32, {4}, {0, 0, 4, 0});
        auto pad_value = ov::opset1::Constant::create(element_type, {}, {0});
        auto pad_op = std::make_shared<ov::opset1::Pad>(result, pads_begin, pads_end, pad_value, ov::op::PadMode::CONSTANT);
        result = pad_op;

        // Step 4: Reshape and transpose weights [C_in, C_out, K] -> [C_out, C_in, K, 1]
        auto weights_4d_shape = ov::opset1::Constant::create(ov::element::i32, {4}, {128, 64, 5, 1});
        auto weights_4d = std::make_shared<ov::opset1::Reshape>(weights, weights_4d_shape, false);
        auto transpose_order = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
        auto weights_transposed = std::make_shared<ov::opset1::Transpose>(weights_4d, transpose_order);

        // Step 5: 2D Convolution
        auto conv = std::make_shared<ov::opset1::Convolution>(result,
                                                              weights_transposed,
                                                              ov::Strides{dilation, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        result = conv;

        // Step 6: Reshape output back to 3D [N, C_out, L_out, 1] -> [N, C_out, L_out]
        // L_out = L + 2*pad - K + 1 = 100 + 2*4 - 5 + 1 = 104 (for stride=1 deconv)
        auto output_3d_shape = ov::opset1::Constant::create(ov::element::i32, {3}, {1, 64, 104});
        auto output_3d = std::make_shared<ov::opset1::Reshape>(result, output_3d_shape, false);

        return std::make_shared<ov::Model>(ov::NodeVector{output_3d}, ov::ParameterVector{input});
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();

        // Build the test model with 1D deconv
        model = createDeconv1D({1, 128, 100}, {128, 64, 5});

        // Build reference model (expected after transformation)
        model_ref = createDeconv1DReference({1, 128, 100}, {128, 64, 5});

        // Register the transformation
        manager.register_pass<Deconv1DDecomposition>();
    }
};

TEST_F(Deconv1DDecompositionTest, BasicStride1) {
    // The transformation and comparison will be performed automatically by TransformationTestsF::TearDown()
    // This test verifies that 1D deconv with stride=1 is properly decomposed
}

class Deconv1DDecomposition4DTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();

        // Build a 4D model that should NOT be transformed (negative test case)
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128, 100, 100});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{128, 64, 5, 5}, {1});
        auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                            weights,
                                                                            ov::Strides{1, 1},
                                                                            ov::CoordinateDiff{0, 0},
                                                                            ov::CoordinateDiff{0, 0},
                                                                            ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input});

        // For negative test case, don't set model_ref - transformation should not change the graph
        // model_ref will be set to clone of original model automatically in TearDown

        manager.register_pass<Deconv1DDecomposition>();
    }
};

TEST_F(Deconv1DDecomposition4DTest, NotAppliedTo4D) {}

// Custom test for dynamic shapes - not using TransformationTestsF since we just want to verify transformation applies
TEST(Deconv1DDynamicTest, DynamicShapes) {
    // Build a dynamic model - should be transformed
    auto model = createDynamicDeconv3D(ov::PartialShape{1, 128, -1}, {128, 64, 5});

    // Verify original model has ConvolutionBackpropData
    auto ops_before = model->get_ordered_ops();
    bool has_deconv_before = std::any_of(ops_before.begin(), ops_before.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::ConvolutionBackpropData>(n.get()); });
    ASSERT_TRUE(has_deconv_before) << "Original model should have ConvolutionBackpropData";

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<Deconv1DDecomposition>();
    manager.run_passes(model);

    // Verify transformation was applied - ConvolutionBackpropData should be replaced
    auto ops_after = model->get_ordered_ops();
    bool has_deconv_after = std::any_of(ops_after.begin(), ops_after.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::ConvolutionBackpropData>(n.get()); });
    EXPECT_FALSE(has_deconv_after) << "ConvolutionBackpropData should be replaced after transformation";

    // Verify expected operations are present
    bool has_convolution = std::any_of(ops_after.begin(), ops_after.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::Convolution>(n.get()); });
    bool has_reshape = std::any_of(ops_after.begin(), ops_after.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::Reshape>(n.get()); });
    EXPECT_TRUE(has_convolution) << "Transformation should create a Convolution node";
    EXPECT_TRUE(has_reshape) << "Transformation should create Reshape nodes";
}

// Custom test for auto_pad with dynamic shapes
TEST(Deconv1DAutoPadDynamicTest, AutoPadWithDynamicShapes) {
    // Build a dynamic model with auto_pad - should be transformed
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{1, 128, ov::Dimension::dynamic()});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, {128, 64, 5}, {1});

    auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
        input,
        weights,
        ov::Strides{2},
        ov::CoordinateDiff{0},     // pads_begin
        ov::CoordinateDiff{0},     // pads_end
        ov::Strides{1},            // dilations
        ov::op::PadType::SAME_UPPER);

    auto model = std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input});

    // Verify original model has ConvolutionBackpropData
    auto ops_before = model->get_ordered_ops();
    bool has_deconv_before = std::any_of(ops_before.begin(), ops_before.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::ConvolutionBackpropData>(n.get()); });
    ASSERT_TRUE(has_deconv_before) << "Original model should have ConvolutionBackpropData";

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<Deconv1DDecomposition>();
    manager.run_passes(model);

    // Verify transformation was applied - ConvolutionBackpropData should be replaced
    auto ops_after = model->get_ordered_ops();
    bool has_deconv_after = std::any_of(ops_after.begin(), ops_after.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::ConvolutionBackpropData>(n.get()); });
    EXPECT_FALSE(has_deconv_after) << "ConvolutionBackpropData should be replaced after transformation";

    // Verify expected operations are present
    bool has_convolution = std::any_of(ops_after.begin(), ops_after.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::Convolution>(n.get()); });
    bool has_pad = std::any_of(ops_after.begin(), ops_after.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::Pad>(n.get()); });
    bool has_shapeof = std::any_of(ops_after.begin(), ops_after.end(),
        [](const auto& n){ return ov::is_type<ov::op::v3::ShapeOf>(n.get()); });

    EXPECT_TRUE(has_convolution) << "Transformation should create a Convolution node";
    EXPECT_TRUE(has_pad) << "Transformation should create a Pad node for dynamic auto_pad";
    EXPECT_TRUE(has_shapeof) << "Transformation should create ShapeOf nodes for dynamic shapes";
}
