// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/deconv_1d_decomposition.hpp"

#include <gtest/gtest.h>

#include <chrono>

#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov::intel_cpu;

struct DeconvTestParams {
    ov::PartialShape input_shape;
    ov::Shape weights_shape;
    size_t stride = 1;
    ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
    bool expect_transformation = true;
    bool with_output_shape = false;  // Test 3-input variant
    std::string test_name;
};

class DeconvDecomposition1DTest : public ::testing::TestWithParam<DeconvTestParams> {
protected:
    std::shared_ptr<ov::Model> model;

    void SetUp() override {
        const auto& p = GetParam();
        const auto rank = p.input_shape.rank().get_length();

        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, p.input_shape);
        auto weights = ov::opset1::Constant::create(ov::element::f32, p.weights_shape, {1});

        // Create strides/pads based on dimensionality (3D vs 4D)
        ov::Strides strides = (rank == 3) ? ov::Strides{p.stride} : ov::Strides{1, 1};
        ov::CoordinateDiff pads = (rank == 3) ? ov::CoordinateDiff{0} : ov::CoordinateDiff{0, 0};
        ov::Strides dilations = (rank == 3) ? ov::Strides{1} : ov::Strides{1, 1};

        std::shared_ptr<ov::Node> deconv;
        if (p.with_output_shape) {
            // Test 3-input variant with explicit output shape (only spatial dimensions)
            ov::Shape output_shape;
            if (rank == 3) {
                // For 3D case, only specify the spatial dimension (L)
                auto in_size = p.input_shape[2].is_static() ? p.input_shape[2].get_length() : 100;
                auto kernel_size = p.weights_shape[2];
                auto out_size = p.stride * (in_size - 1) + kernel_size;
                output_shape = {out_size};  // Only spatial dimension
            } else {
                output_shape = {100, 100};  // 4D case - two spatial dimensions
            }
            auto output_shape_const =
                ov::opset1::Constant::create(ov::element::i64, {output_shape.size()}, output_shape);
            deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                           weights,
                                                                           output_shape_const,
                                                                           strides,
                                                                           pads,
                                                                           pads,
                                                                           dilations,
                                                                           p.pad_type);
        } else {
            // Test 2-input variant
            deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                           weights,
                                                                           strides,
                                                                           pads,
                                                                           pads,
                                                                           dilations,
                                                                           p.pad_type);
        }

        model = std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input});
    }

    template <typename T>
    [[nodiscard]] bool has_op_of_type() const {
        auto ops = model->get_ordered_ops();
        return std::any_of(ops.begin(), ops.end(), [](const auto& n) {
            return ov::is_type<T>(n.get());
        });
    }
};

TEST_P(DeconvDecomposition1DTest, TransformationTest) {
    const auto& p = GetParam();

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<Deconv1DDecomposition>();

    // Measure time
    auto start = std::chrono::steady_clock::now();
    manager.run_passes(model);
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    EXPECT_LT(duration, 1000) << "Transformation took " << duration << "ms";

    // Verify transformation results
    if (p.expect_transformation) {
        EXPECT_FALSE(has_op_of_type<ov::op::v1::ConvolutionBackpropData>()) << "Deconv should be replaced";
        EXPECT_TRUE(has_op_of_type<ov::op::v1::Convolution>()) << "Should have Convolution";
        EXPECT_TRUE(has_op_of_type<ov::op::v1::Reshape>()) << "Should have Reshape";
    } else {
        EXPECT_TRUE(has_op_of_type<ov::op::v1::ConvolutionBackpropData>()) << "Deconv should NOT be replaced";
    }
}

INSTANTIATE_TEST_SUITE_P(DeconvDecomposition1DTests,
                         DeconvDecomposition1DTest,
                         ::testing::Values(
                             // 3D Static tests (2-input variant)
                             DeconvTestParams{ov::PartialShape{1, 128, 100},
                                              ov::Shape{128, 64, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              false,
                                              "Static3D_Stride1"},
                             // 3D Static test (3-input variant with output shape)
                             DeconvTestParams{ov::PartialShape{1, 128, 100},
                                              ov::Shape{128, 64, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              true,
                                              "Static3D_WithOutputShape"},
                             // 3D Dynamic tests
                             DeconvTestParams{ov::PartialShape{1, 2, -1},
                                              ov::Shape{2, 2, 3},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              false,
                                              "Dynamic3D_Simple"},
                             DeconvTestParams{ov::PartialShape{1, 128, -1},
                                              ov::Shape{128, 64, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              false,
                                              "Dynamic3D_Large"},
                             DeconvTestParams{ov::PartialShape{1, 128, ov::Dimension::dynamic()},
                                              ov::Shape{128, 64, 5},
                                              2,
                                              ov::op::PadType::SAME_UPPER,
                                              true,
                                              false,
                                              "Dynamic3D_AutoPad_Stride2"},
                             // 3D Dynamic test (3-input variant)
                             DeconvTestParams{ov::PartialShape{1, 128, -1},
                                              ov::Shape{128, 64, 5},
                                              2,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              true,
                                              "Dynamic3D_WithOutputShape_Stride2"},
                             // 4D Negative test
                             DeconvTestParams{ov::PartialShape{1, 128, 100, 100},
                                              ov::Shape{128, 64, 5, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              false,  // Should NOT transform
                                              false,
                                              "Static4D_NotTransformed"}),
                         [](const testing::TestParamInfo<DeconvTestParams>& info) {
                             return info.param.test_name;
                         });
