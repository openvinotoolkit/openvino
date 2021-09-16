// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/compress_quantize_weights.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, CompressQuantizeWeightsI4) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = opset8::Constant::create(element::f32, Shape{2, 3, 1, 1}, {-1, 2, 3, 4, 5, 11});
        auto input_low = opset8::Constant::create(element::f32, Shape{}, {0});
        auto input_high = opset8::Constant::create(element::f32, Shape{}, {10});
        auto output_low = opset8::Constant::create(element::f32, Shape{}, {-1});
        auto output_high = opset8::Constant::create(element::f32, Shape{}, {4});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 16);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::CompressQuantizeWeights>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = opset8::Constant::create(element::i4, Shape{2, 3, 1, 1}, {-8, -5, -4, -2, 0, 7});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {0.5});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {2});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CompressQuantizeWeightsI8) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = opset8::Constant::create(element::f32, Shape{2, 4, 1, 1}, {-1, 0, 1, 2, 3, 4, 5, 11});
        auto input_low = opset8::Constant::create(element::f32, Shape{}, {1});
        auto input_high = opset8::Constant::create(element::f32, Shape{}, {9});
        auto output_low = opset8::Constant::create(element::f32, Shape{}, {-2});
        auto output_high = opset8::Constant::create(element::f32, Shape{}, {6});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::CompressQuantizeWeights>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = opset8::Constant::create(element::i8, Shape{2, 4, 1, 1}, {-128, -128, -128, -96, -64, -32, 0, 127});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {1});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {3});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeCompressQuantizeWeightsNonConstantInput) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 1, 1});
    auto input_low = opset8::Constant::create(element::f32, Shape{}, {1});
    auto input_high = opset8::Constant::create(element::f32, Shape{}, {9});
    auto output_low = opset8::Constant::create(element::f32, Shape{}, {-2});
    auto output_high = opset8::Constant::create(element::f32, Shape{}, {6});
    auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
    f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<pass::CompressQuantizeWeights>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    f_ref = clone_function(*f);

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
