// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "utils/cpu_test_utils.hpp"

namespace {
using namespace CPUTestUtils;

// Regression test for #32070
// Verifies that getScalesAndShifts correctly handles memory with blocked layouts
// by using actual allocated size instead of incorrectly calculating element count
class ScalesShiftsOOBTest : public SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector params;
        auto input = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f32, 
            ov::Shape{1, 1, 240, 320});
        params.push_back(input);

        // Conv layer
        auto weights1 = ov::test::utils::make_constant(
            ov::element::f32,
            ov::Shape{24, 1, 3, 3});
        auto conv1 = std::make_shared<ov::op::v1::Convolution>(
            input,
            weights1,
            ov::Strides{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::Strides{1, 1});

        auto slope1 = ov::test::utils::make_constant(
            ov::element::f32,
            ov::Shape{24},
            std::vector<float>(24, 0.1f));
        auto prelu1 = std::make_shared<ov::op::v0::PRelu>(conv1, slope1);

        auto weights2 = ov::test::utils::make_constant(
            ov::element::f32,
            ov::Shape{24, 1, 1, 3, 3});
        auto group_conv = std::make_shared<ov::op::v1::GroupConvolution>(
            prelu1,
            weights2,
            ov::Strides{2, 2},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});

        auto slope2 = ov::test::utils::make_constant(
            ov::element::f32,
            ov::Shape{24},
            std::vector<float>(24, 0.1f));
        auto prelu2 = std::make_shared<ov::op::v0::PRelu>(group_conv, slope2);

        auto weights3 = ov::test::utils::make_constant(
            ov::element::f32,
            ov::Shape{40, 24, 1, 1});
        auto conv3 = std::make_shared<ov::op::v1::Convolution>(
            prelu2,
            weights3,
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});

        auto slope3 = ov::test::utils::make_constant(
            ov::element::f32,
            ov::Shape{40},
            std::vector<float>(40, 0.1f));
        auto prelu3 = std::make_shared<ov::op::v0::PRelu>(conv3, slope3);

        auto weights4 = ov::test::utils::make_constant(
            ov::element::f32,
            ov::Shape{40, 1, 1, 3, 3});
        auto group_conv2 = std::make_shared<ov::op::v1::GroupConvolution>(
            prelu3,
            weights4,
            ov::Strides{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::Strides{1, 1});

        auto slope4 = ov::test::utils::make_constant(
            ov::element::f32,
            ov::Shape{40},
            std::vector<float>(40, 0.1f));
        auto prelu4 = std::make_shared<ov::op::v0::PRelu>(group_conv2, slope4);

        function = std::make_shared<ov::Model>(
            ov::NodeVector{prelu4},
            params,
            "ScalesShiftsOOBTest");
    }
};

TEST_F(ScalesShiftsOOBTest, CompileAndInfer) {
    // Before the fix, this would read beyond allocated memory when processing
    // PReLU slope constants in blocked layouts
    run();
}

}  // namespace
