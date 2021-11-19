// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <compress_quantize_weights.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


static void test_compress_quantize_weights(const Shape& shape, const std::vector<float>& weights,
                                           float in_low, float in_high, float out_low, float out_high, size_t levels,
                                           const element::Type_t& expected_type, const std::vector<float>& expected_weights,
                                           float scale_val, float zero_point_val) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = opset8::Constant::create(element::f32, shape, weights);
        auto input_low = opset8::Constant::create(element::f32, Shape{}, {in_low});
        auto input_high = opset8::Constant::create(element::f32, Shape{}, {in_high});
        auto output_low = opset8::Constant::create(element::f32, Shape{}, {out_low});
        auto output_high = opset8::Constant::create(element::f32, Shape{}, {out_high});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::CompressQuantizeWeights>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = opset8::Constant::create(expected_type, shape, expected_weights);
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {scale_val});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {zero_point_val});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});
    }
    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CompressQuantizeWeightsI4) {
    test_compress_quantize_weights(Shape{2, 3, 1, 1},
                                   {-1, 2, 3, 4, 5, 11}, // weights
                                   0, // input low
                                   10, // input high
                                   -1, // output low
                                   4, // output high
                                   3, // levels
                                   element::i4, // expected_type
                                   {-1, -1, 0, 0, 0, 0}, // expected_weights
                                   2.5, // scale
                                   -1.1); // zero_point
    test_compress_quantize_weights(Shape{2, 3, 1, 1},
                                   {-1, 2, 3, 4, 5, 11}, // weights
                                   0, // input low
                                   10, // input high
                                   -1, // output low
                                   4, // output high
                                   16, // levels
                                   element::i4, // expected_type
                                   {-8, -5, -4, -2, 0, 7}, // expected_weights
                                   0.333333, // scale
                                   -5); // zero_point
}

TEST(TransformationTests, CompressQuantizeWeightsI8) {
    test_compress_quantize_weights(Shape{2, 4, 1, 1},
                                   {-1, 0, 1, 2, 3, 4, 5, 11}, // weights
                                   1, // input low
                                   9, // input high
                                   -2, // output low
                                   6, // output high
                                   17, // levels
                                   element::i8, // expected_type
                                   {-8, -8, -8, -6, -4, -2, 0, 7}, // expected_weights
                                   0.5, // scale
                                   -4.5); // zero_point
    test_compress_quantize_weights(Shape{2, 4, 1, 1},
                                   {-1, 0, 1, 2, 3, 4, 5, 11}, // weights
                                   1, // input low
                                   9, // input high
                                   -2, // output low
                                   6, // output high
                                   256, // levels
                                   element::i8, // expected_type
                                   {-128, -128, -128, -96, -64, -32, 0, 127}, // expected_weights
                                   0.0313725, // scale
                                   -64.25); // zero_point
}

TEST(TransformationTests, CompressQuantizeWeightsWithDequantizationSubgraph) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = opset8::Constant::create(element::f32, Shape{2, 4, 1, 1}, {-1, 0, 1, 2, 3, 4, 5, 11});
        auto input_low = opset8::Constant::create(element::f32, Shape{}, {1});
        auto input_high = opset8::Constant::create(element::f32, Shape{}, {9});
        auto output_low = opset8::Constant::create(element::f32, Shape{}, {-128});
        auto output_high = opset8::Constant::create(element::f32, Shape{}, {127});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
        auto convert = std::make_shared<opset8::Convert>(fq, element::i8);
        auto second_convert = std::make_shared<opset8::Convert>(convert, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {10.0 / 255});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {2 - 255.0 / 10});
        auto sub = std::make_shared<opset8::Subtract>(second_convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);

        f = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::CompressQuantizeWeights>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = opset8::Constant::create(element::i8, Shape{2, 4, 1, 1}, {-128, -128, -128, -96, -64, -32, 0, 127});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {10.0 / 255});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {2 - 255.0 / 10});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CompressQuantizeWeightsWithZeroPointOptimizer) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.144816, 0.0858578, 0.110928});
        auto input_low = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.402659, -0.383148, -0.34054});
        auto input_high = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.399513, 0.380155, 0.33788});
        auto output_low = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.402659, -0.383148, -0.34054});
        auto output_high = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.399513, 0.380155, 0.33788});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        auto gr = m.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::CompressQuantizeWeights>();
        gr->add_matcher<ngraph::pass::ZeroPointOptimizer>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = opset8::Constant::create(element::i8, Shape{3, 1, 1, 1}, {-46, 29, 42});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.00314577, 0.00299335, 0.00266047});
        auto mul = std::make_shared<opset8::Multiply>(convert, scale);
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});
    }
    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeCompressQuantizeWeightsWithZeroPointOptimizer) {
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
        auto gr = m.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::CompressQuantizeWeights>();
        gr->add_matcher<ngraph::pass::ZeroPointOptimizer>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = opset8::Constant::create(element::i8, Shape{2, 4, 1, 1}, {-128, -128, -128, -96, -64, -32, 0, 127});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {0.0313725});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {-64.25});
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
