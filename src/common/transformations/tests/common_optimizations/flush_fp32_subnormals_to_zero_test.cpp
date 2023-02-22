// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/flush_fp32_subnormals_to_zero.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph/pass/visualize_tree.hpp"

using namespace testing;
using namespace ngraph;
using namespace std;

namespace {
union FloatIntUnion {
    uint32_t u;
    float f;
};
FloatIntUnion maximum_subnorm_val = {0x007fffff};  // = 2^−126 * (1 - 2^−23) ~= 1.1754942107e-38f
FloatIntUnion minimum_subnorm_val = {0x00000001};  // = 2^−149 ~= 1.4012984643e-45f
FloatIntUnion minimum_norm_val = {0x00800000};     // = 2^−126 ~= 1.1754943508-38f
}  // namespace

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_max_subnorm) {
    float subnormal_val = maximum_subnorm_val.f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           5.0f,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights =
            ov::opset8::Constant::create(ov::element::f32,
                                         ov::Shape{1, 3, 4, 1},
                                         {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        function_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_min_subnorm) {
    float subnormal_val = minimum_subnorm_val.f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           5.0f,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights =
            ov::opset8::Constant::create(ov::element::f32,
                                         ov::Shape{1, 3, 4, 1},
                                         {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        function_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_arbitrary_subnorm) {
    float subnormal_val = 2.0e-44f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           5.0f,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights =
            ov::opset8::Constant::create(ov::element::f32,
                                         ov::Shape{1, 3, 4, 1},
                                         {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        function_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_max_neg_subnorm) {
    float subnormal_val = -maximum_subnorm_val.f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           5.0f,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights =
            ov::opset8::Constant::create(ov::element::f32,
                                         ov::Shape{1, 3, 4, 1},
                                         {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        function_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_min_neg_subnorm) {
    float subnormal_val = -minimum_subnorm_val.f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           5.0f,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights =
            ov::opset8::Constant::create(ov::element::f32,
                                         ov::Shape{1, 3, 4, 1},
                                         {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        function_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_arbitrary_neg_subnorm) {
    float subnormal_val = -2.0e-45f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           5.0f,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val,
                                                           subnormal_val});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights =
            ov::opset8::Constant::create(ov::element::f32,
                                         ov::Shape{1, 3, 4, 1},
                                         {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        function_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, test_flush_fp32_subnorm_to_zero_arbitrary_norm) {
    // minimum normalized val should not be flushed to zero
    float normal_val = minimum_norm_val.f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           5.0f,
                                                           normal_val,
                                                           normal_val,
                                                           normal_val,
                                                           normal_val,
                                                           normal_val,
                                                           normal_val});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        function = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::FlushFP32SubnormalsToZero>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           5.0f,
                                                           normal_val,
                                                           normal_val,
                                                           normal_val,
                                                           normal_val,
                                                           normal_val,
                                                           normal_val});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        function_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
