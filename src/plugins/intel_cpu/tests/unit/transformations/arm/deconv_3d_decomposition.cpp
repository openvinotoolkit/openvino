// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/deconv_3d_decomposition.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov::intel_cpu;

class Deconv3DDecompositionTest : public testing::Test {
protected:
    static std::shared_ptr<ov::Model> createDeconv3D(const ov::Shape& input_shape,
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

    static std::shared_ptr<ov::Model> createDynamicWeightsDeconv3D(const ov::Shape& input_shape,
                                                                   const ov::PartialShape& weights_shape) {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
        auto weights = std::make_shared<ov::opset1::Parameter>(ov::element::f32, weights_shape);

        auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                            weights,
                                                                            ov::Strides{1},
                                                                            ov::CoordinateDiff{0},
                                                                            ov::CoordinateDiff{0},
                                                                            ov::Strides{1});

        return std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input, weights});
    }

    static void applyTransformationAndVerify(const std::shared_ptr<ov::Model>& model,
                                             bool should_transform = true,
                                             bool check_rt = true) {
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<Deconv3DDecomposition>();
        m.run_passes(model);

        if (check_rt) {
            ASSERT_NO_THROW(check_rt_info(model));
        }

        auto ops = model->get_ordered_ops();
        bool has_convolution = std::any_of(ops.begin(), ops.end(),
            [](const auto& n){ return ov::is_type<ov::opset1::Convolution>(n.get()); });
        bool has_deconv = std::any_of(ops.begin(), ops.end(),
            [](const auto& n){ return ov::is_type<ov::opset1::ConvolutionBackpropData>(n.get()); });

        if (should_transform) {
            ASSERT_TRUE(has_convolution) << "Transformation should replace deconv with conv";
            ASSERT_FALSE(has_deconv) << "Original deconv should be removed";
        } else {
            ASSERT_FALSE(has_convolution) << "Transformation should not be applied";
            ASSERT_TRUE(has_deconv) << "Original deconv should remain";
        }
    }
};
TEST_F(Deconv3DDecompositionTest, BasicStride1) {
    // Input: [batch=1, channels=128, length=100]
    // Weights: [in_channels=128, out_channels=64, kernel=5]
    // Expected output: [1, 64, 104]
    auto model = createDeconv3D({1, 128, 100}, {128, 64, 5});
    applyTransformationAndVerify(model);
}

TEST_F(Deconv3DDecompositionTest, Stride4WithUpsampling) {
    // Real-world example from demucs model
    // Input: [1, 384, 1344], stride=4, kernel=8
    // After upsampling: [1, 384, 5373]
    // Expected output: [1, 384, 5380]
    auto model = createDeconv3D({1, 384, 1344}, {384, 384, 8}, 4);
    applyTransformationAndVerify(model);

    auto ops = model->get_ordered_ops();
    bool has_scatter = std::any_of(ops.begin(), ops.end(),
        [](const auto& n){ return ov::is_type<ov::op::v3::ScatterUpdate>(n.get()); });
    ASSERT_TRUE(has_scatter) << "ScatterUpdate should be used for stride > 1";
}

TEST_F(Deconv3DDecompositionTest, WithPadding) {
    // Input: [1, 64, 50], kernel=7, stride=2, pad=2
    // Padding in a decomposed version: pad_left = 7-1-2 = 4, pad_right = 4
    auto model = createDeconv3D({1, 64, 50}, {64, 32, 7}, 2, 2, 2);
    applyTransformationAndVerify(model);
}

TEST_F(Deconv3DDecompositionTest, NotAppliedTo4D) {
    // 4D input should not be transformed
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 128, 100, 100});
    auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{128, 64, 5, 5}, {1});

    auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                        weights,
                                                                        ov::Strides{1, 1},
                                                                        ov::CoordinateDiff{0, 0},
                                                                        ov::CoordinateDiff{0, 0},
                                                                        ov::Strides{1, 1});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input});
    applyTransformationAndVerify(model, false);  // should NOT transform
}

TEST_F(Deconv3DDecompositionTest, DifferentDataTypes) {
    auto model =
        createDeconv3D({1, 96, 21499}, {96, 96, 8}, 4, 0, 0, 1, ov::op::PadType::EXPLICIT, 0, ov::element::f16);
    applyTransformationAndVerify(model);

    auto ops = model->get_ordered_ops();
    for (auto& op : ops) {
        if (ov::is_type<ov::opset1::Convolution>(op.get())) {
            auto conv = ov::as_type_ptr<ov::opset1::Convolution>(op);
            ASSERT_EQ(conv->get_output_element_type(0), ov::element::f16);
        }
    }
}

TEST_F(Deconv3DDecompositionTest, OutputPadding) {
    // Output padding affects the final size
    // pad_right = kernel - 1 - pad_end + output_padding
    auto model = createDeconv3D({1, 48, 85995}, {48, 48, 8}, 4, 0, 0, 1, ov::op::PadType::EXPLICIT, 2);
    applyTransformationAndVerify(model);
}

TEST_F(Deconv3DDecompositionTest, WithDilation) {
    // Dilation affects the effective kernel size
    auto model = createDeconv3D({1, 128, 100}, {128, 64, 5}, 1, 0, 0, 2);
    applyTransformationAndVerify(model);

    auto ops = model->get_ordered_ops();
    for (auto& op : ops) {
        if (ov::is_type<ov::opset1::Convolution>(op.get())) {
            auto conv = ov::as_type_ptr<ov::opset1::Convolution>(op);
            ASSERT_EQ(conv->get_dilations(), ov::Strides({2, 1}));
        }
    }
}

TEST_F(Deconv3DDecompositionTest, DynamicShapes) {
    // Input with dynamic length dimension
    auto model = createDynamicDeconv3D(ov::PartialShape{1, 128, -1}, {128, 64, 5});
    applyTransformationAndVerify(model, true, false);  // skip RT info check for dynamic shapes
}

TEST_F(Deconv3DDecompositionTest, DynamicShapesWithPadding) {
    auto model = createDynamicDeconv3D(ov::PartialShape{1, 64, -1}, {64, 32, 7}, 1, 2, 2);
    applyTransformationAndVerify(model, true, false);  // skip RT info check for dynamic shapes
}

TEST_F(Deconv3DDecompositionTest, DynamicWeights) {
    auto model = createDynamicWeightsDeconv3D({1, 64, 100}, {64, 32, -1});
    applyTransformationAndVerify(model, true, false);  // skip RT info check for dynamic shapes

    // Verify dynamic padding is present
    auto ops = model->get_ordered_ops();
    bool has_dynamic_pad = std::any_of(ops.begin(), ops.end(),
        [](const auto& n){ return ov::is_type<ov::op::v1::Pad>(n.get()); });
    ASSERT_TRUE(has_dynamic_pad) << "Dynamic padding should be present for dynamic kernel size";
}

TEST_F(Deconv3DDecompositionTest, AutoPadSame) {
    // SAME padding means output_size = input_size * stride
    auto model = createDeconv3D({1, 128, 100}, {128, 64, 5}, 1, 0, 0, 1, ov::op::PadType::SAME_UPPER);
    applyTransformationAndVerify(model);
}

TEST_F(Deconv3DDecompositionTest, AutoPadValid) {
    // VALID padding means no padding
    auto model = createDeconv3D({1, 128, 100}, {128, 64, 5}, 1, 0, 0, 1, ov::op::PadType::VALID);
    applyTransformationAndVerify(model);
}

TEST_F(Deconv3DDecompositionTest, LargeDimensions) {
    auto model = createDeconv3D({1, 768, 17280}, {768, 768, 16}, 4);
    applyTransformationAndVerify(model);

    auto output = model->get_results()[0];
    auto output_shape = output->get_output_partial_shape(0);
    ASSERT_TRUE(output_shape.is_static());

    // Expected: (17,280-1) * 4 + 1 = 69,117
    // Then padded: 69,117 + 15 + 15 = 69,147
    // Then after conv with kernel 16: 69,147-16 + 1 = 69,132
    ASSERT_EQ(output_shape.to_shape(), ov::Shape({1, 768, 69132}));
}
