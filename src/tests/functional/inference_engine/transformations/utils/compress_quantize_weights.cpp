// Copyright (C) 2018-2022 Intel Corporation
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


struct CompressQuantizeWeightsParams {
    Shape shape;
    std::vector<float> weights;
    float in_low;
    float in_high;
    float out_low;
    float out_high;
    size_t levels;
    element::Type_t expected_type;
    std::vector<float> expected_weights;
    float scale_val;
    float zero_point_val;
};

class CompressQuantizeWeightsTests
        : public testing::WithParamInterface<CompressQuantizeWeightsParams>,
          public TransformationTestsF {
    void SetUp() override {
        TransformationTestsF::SetUp();
        auto param = GetParam();
        {
            auto data = opset8::Constant::create(element::f32, param.shape, param.weights);
            auto input_low = opset8::Constant::create(element::f32, Shape{}, {param.in_low});
            auto input_high = opset8::Constant::create(element::f32, Shape{}, {param.in_high});
            auto output_low = opset8::Constant::create(element::f32, Shape{}, {param.out_low});
            auto output_high = opset8::Constant::create(element::f32, Shape{}, {param.out_high});
            auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, param.levels);
            function = std::make_shared<Function>(fq, ParameterVector{});
        }

        manager.register_pass<pass::CompressQuantizeWeights>();

        {
            auto data = opset8::Constant::create(param.expected_type, param.shape, param.expected_weights);
            auto convert = std::make_shared<opset8::Convert>(data, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {param.scale_val});
            auto zero_point = opset8::Constant::create(element::f32, Shape{}, {param.zero_point_val});
            auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
            auto mul = std::make_shared<opset8::Multiply>(sub, scale);
            function_ref = std::make_shared<Function>(mul, ParameterVector{});
        }
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    }
};

TEST_P(CompressQuantizeWeightsTests, FusionTest) {
}

static std::vector<CompressQuantizeWeightsParams> params = {
    {Shape{2, 3, 1, 1}, {-1, 2, 3, 4, 5, 11}, 0, 10, -1, 5, 3, element::i4, {-1, -1, 0, 0, 0, 1}, 3, -0.666667},
    {Shape{2, 3, 1, 1}, {-1, 2, 3, 4, 5, 11}, 0, 10, -1, 4, 16, element::i4, {-8, -5, -4, -2, 0, 7}, 0.333333, -5},
    {Shape{2, 4, 1, 1}, {-1, 0, 1, 2, 3, 4, 5, 11}, 1, 9, -2, 6, 17, element::i8, {-8, -8, -8, -6, -4, -2, 0, 8}, 0.5, -4},
    {Shape{2, 4, 1, 1}, {-1, 0, 1, 2, 3, 4, 5, 11}, 1, 9, -2, 6, 256, element::i8, {-128, -128, -128, -96, -64, -32, 0, 127}, 0.0313725, -64.25},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests, CompressQuantizeWeightsTests, ::testing::ValuesIn(params));


TEST_F(TransformationTestsF, CompressQuantizeWeightsWithDequantizationSubgraph) {
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

        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});

        manager.register_pass<pass::CompressQuantizeWeights>();
    }
    {
        auto data = opset8::Constant::create(element::i8, Shape{2, 4, 1, 1}, {-128, -128, -128, -96, -64, -32, 0, 127});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {10.0 / 255});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {2 - 255.0 / 10});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        function_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, CompressQuantizeWeightsWithZeroPointOptimizer) {
    {
        auto data = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.144816, 0.0858578, 0.110928});
        auto input_low = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.402659, -0.383148, -0.34054});
        auto input_high = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.399513, 0.380155, 0.33788});
        auto output_low = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.402659, -0.383148, -0.34054});
        auto output_high = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.399513, 0.380155, 0.33788});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
        function = std::make_shared<Function>(NodeVector{fq}, ParameterVector{});

        manager.register_pass<pass::CompressQuantizeWeights>();
        manager.register_pass<pass::ZeroPointOptimizer>();
    }

    {
        auto data = opset8::Constant::create(element::i8, Shape{3, 1, 1, 1}, {-46, 29, 42});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.00314577, 0.00299335, 0.00266047});
        auto mul = std::make_shared<opset8::Multiply>(convert, scale);
        function_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, NegativeCompressQuantizeWeightsWithZeroPointOptimizer) {
    {
        auto data = opset8::Constant::create(element::f32, Shape{2, 4, 1, 1}, {-1, 0, 1, 2, 3, 4, 5, 11});
        auto input_low = opset8::Constant::create(element::f32, Shape{}, {1});
        auto input_high = opset8::Constant::create(element::f32, Shape{}, {9});
        auto output_low = opset8::Constant::create(element::f32, Shape{}, {-2});
        auto output_high = opset8::Constant::create(element::f32, Shape{}, {6});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
        function = std::make_shared<Function>(NodeVector{fq}, ParameterVector{});

        manager.register_pass<pass::CompressQuantizeWeights>();
        manager.register_pass<pass::ZeroPointOptimizer>();
    }
    {
        auto data = opset8::Constant::create(element::i8, Shape{2, 4, 1, 1}, {-128, -128, -128, -96, -64, -32, 0, 127});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {0.0313725});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {-64.25});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        function_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, NegativeCompressQuantizeWeightsNonConstantInput) {
    auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 1, 1});
    auto input_low = opset8::Constant::create(element::f32, Shape{}, {1});
    auto input_high = opset8::Constant::create(element::f32, Shape{}, {9});
    auto output_low = opset8::Constant::create(element::f32, Shape{}, {-2});
    auto output_high = opset8::Constant::create(element::f32, Shape{}, {6});
    auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
    function = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});

    manager.register_pass<pass::CompressQuantizeWeights>();
    manager.register_pass<pass::ZeroPointOptimizer>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
