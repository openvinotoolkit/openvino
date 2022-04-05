// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/common_optimizations/conv_to_binary_conv.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, ConvToBinaryConvOutputLowZeroOutputHighOne) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, op::PadType::EXPLICIT);

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvToBinaryConv>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        uint8_t weights_val = 6;
        auto weights = std::make_shared<opset5::Constant>(element::u1, Shape{1, 3, 1, 1}, &weights_val);
        auto conv = std::make_shared<opset5::BinaryConvolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0},
                Strides{1, 1}, opset5::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT, -1, op::PadType::EXPLICIT);
        auto add = std::make_shared<opset5::Add>(conv, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.7f}));
        auto mul = std::make_shared<opset5::Multiply>(add, opset5::Constant::create(element::f32, Shape{}, {0.2f}));

        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvToBinaryConvOutputLowMinusOneOutputHighOne) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, op::PadType::EXPLICIT);

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvToBinaryConv>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        uint8_t weights_val = 6;
        auto weights = std::make_shared<opset5::Constant>(element::u1, Shape{1, 3, 1, 1}, &weights_val);
        auto conv = std::make_shared<opset5::BinaryConvolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0},
                Strides{1, 1}, opset5::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT, 0, op::PadType::EXPLICIT);

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeConvToBinaryConvInvalidWeights) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 2, 3});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, op::PadType::EXPLICIT);

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvToBinaryConv>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 2, 3});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, op::PadType::EXPLICIT);

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeConvToBinaryConvInvalidLevels) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 3);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, op::PadType::EXPLICIT);

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvToBinaryConv>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 3);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, op::PadType::EXPLICIT);

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeConvToBinaryConvOutputLowHigh) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-2.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, op::PadType::EXPLICIT);

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvToBinaryConv>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-2.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0},
                CoordinateDiff{0, 0}, Strides{1, 1}, op::PadType::EXPLICIT);

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
