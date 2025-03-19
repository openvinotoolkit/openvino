// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/conv_to_binary_conv.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST(TransformationTests, ConvToBinaryConvOutputLowZeroOutputHighOne) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1},
                                                          op::PadType::EXPLICIT);

        f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvToBinaryConv>();
        m.register_pass<ov::pass::ConstantFolding>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        uint8_t weights_val = 6;
        auto weights = std::make_shared<opset5::Constant>(element::u1, Shape{1, 3, 1, 1}, &weights_val);
        auto conv =
            std::make_shared<opset5::BinaryConvolution>(act_fq,
                                                        weights,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1},
                                                        opset5::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
                                                        -1.0f,
                                                        op::PadType::EXPLICIT);
        auto add = std::make_shared<opset5::Add>(conv, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.7f}));
        auto mul = std::make_shared<opset5::Multiply>(add, opset5::Constant::create(element::f32, Shape{}, {0.2f}));

        f_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvToBinaryConvOutputLowMinusOneOutputHighOne) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1},
                                                          op::PadType::EXPLICIT);

        f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvToBinaryConv>();
        m.register_pass<ov::pass::ConstantFolding>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        uint8_t weights_val = 6;
        auto weights = std::make_shared<opset5::Constant>(element::u1, Shape{1, 3, 1, 1}, &weights_val);
        auto conv =
            std::make_shared<opset5::BinaryConvolution>(act_fq,
                                                        weights,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1},
                                                        opset5::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
                                                        0.0f,
                                                        op::PadType::EXPLICIT);

        f_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeConvToBinaryConvInvalidWeights) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 2, 3});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1},
                                                          op::PadType::EXPLICIT);

        f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvToBinaryConv>();
        m.register_pass<ov::pass::ConstantFolding>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 2, 3});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1},
                                                          op::PadType::EXPLICIT);

        f_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeConvToBinaryConvInvalidLevels) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 3);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1},
                                                          op::PadType::EXPLICIT);

        f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvToBinaryConv>();
        m.register_pass<ov::pass::ConstantFolding>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 3);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1},
                                                          op::PadType::EXPLICIT);

        f_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeConvToBinaryConvOutputLowHigh) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-2.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1},
                                                          op::PadType::EXPLICIT);

        f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvToBinaryConv>();
        m.register_pass<ov::pass::ConstantFolding>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-2.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1},
                                                          op::PadType::EXPLICIT);

        f_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
