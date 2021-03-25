// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/common_optimizations/binarize_weights.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, BinarizeWeightsActivationsOutputLowZero) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights, weights_in_low, weights_in_high, weights_out_low, weights_out_high, 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights_fq, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::BinarizeWeights>();
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
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto mul = std::make_shared<opset5::Multiply>(conv, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.7f}));
        auto mul2 = std::make_shared<opset5::Multiply>(mul, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.2f}));

        f_ref = std::make_shared<Function>(NodeVector{mul2}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BinarizeWeightsActivationsOutputLowNegative) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.7f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights, weights_in_low, weights_in_high, weights_out_low, weights_out_high, 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights_fq, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::BinarizeWeights>();
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
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto mul = std::make_shared<opset5::Multiply>(conv, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.7f}));
        auto mul2 = std::make_shared<opset5::Multiply>(mul, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.2f}));

        f_ref = std::make_shared<Function>(NodeVector{mul2}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeBinarizeWeightsInvalidLevels) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.7f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 3);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights, weights_in_low, weights_in_high, weights_out_low, weights_out_high, 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights_fq, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::BinarizeWeights>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.7f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 3);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights, weights_in_low, weights_in_high, weights_out_low, weights_out_high, 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights_fq, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeBinarizeWeightsInvalidActivationsOutputLowHigh) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights, weights_in_low, weights_in_high, weights_out_low, weights_out_high, 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights_fq, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::BinarizeWeights>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights, weights_in_low, weights_in_high, weights_out_low, weights_out_high, 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights_fq, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeBinarizeWeightsInvalidOutputLowHigh) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights, weights_in_low, weights_in_high, weights_out_low, weights_out_high, 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights_fq, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::BinarizeWeights>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq = std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights, weights_in_low, weights_in_high, weights_out_low, weights_out_high, 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq, weights_fq, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
