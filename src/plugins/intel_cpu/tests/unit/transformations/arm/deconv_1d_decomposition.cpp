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
    std::string test_name;
};

class Deconv1DTest : public ::testing::TestWithParam<DeconvTestParams> {
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

        auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                            weights,
                                                                            strides,
                                                                            pads,
                                                                            pads,
                                                                            dilations,
                                                                            p.pad_type);

        model = std::make_shared<ov::Model>(ov::NodeVector{deconv}, ov::ParameterVector{input});
    }

    template <typename T>
    bool has_op_of_type() const {
        auto ops = model->get_ordered_ops();
        return std::any_of(ops.begin(), ops.end(), [](const auto& n) {
            return ov::is_type<T>(n.get());
        });
    }
};

TEST_P(Deconv1DTest, TransformationTest) {
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

INSTANTIATE_TEST_SUITE_P(Deconv1DTests,
                         Deconv1DTest,
                         ::testing::Values(
                             // 3D Static tests
                             DeconvTestParams{ov::PartialShape{1, 128, 100},
                                              ov::Shape{128, 64, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              "Static3D_Stride1"},
                             // 3D Dynamic tests
                             DeconvTestParams{ov::PartialShape{1, 2, -1},
                                              ov::Shape{2, 2, 3},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              "Dynamic3D_Simple"},
                             DeconvTestParams{ov::PartialShape{1, 128, -1},
                                              ov::Shape{128, 64, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              true,
                                              "Dynamic3D_Large"},
                             DeconvTestParams{ov::PartialShape{1, 128, ov::Dimension::dynamic()},
                                              ov::Shape{128, 64, 5},
                                              2,
                                              ov::op::PadType::SAME_UPPER,
                                              true,
                                              "Dynamic3D_AutoPad_Stride2"},
                             // 4D Negative test
                             DeconvTestParams{ov::PartialShape{1, 128, 100, 100},
                                              ov::Shape{128, 64, 5, 5},
                                              1,
                                              ov::op::PadType::EXPLICIT,
                                              false,  // Should NOT transform
                                              "Static4D_NotTransformed"}),
                         [](const testing::TestParamInfo<DeconvTestParams>& info) {
                             return info.param.test_name;
                         });
