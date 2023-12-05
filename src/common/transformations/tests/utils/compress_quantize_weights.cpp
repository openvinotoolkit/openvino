// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compress_quantize_weights.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

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
    bool fuse_zero_point;
};

class CompressQuantizeWeightsTests
    : public testing::WithParamInterface<std::tuple<CompressQuantizeWeightsParams, element::Type>>,
      public TransformationTestsF {
    void SetUp() override {
        TransformationTestsF::SetUp();
        CompressQuantizeWeightsParams param;
        ov::element::Type data_prc;
        std::tie(param, data_prc) = GetParam();
        {
            std::shared_ptr<Node> data = opset8::Constant::create(data_prc, param.shape, param.weights);
            if (data_prc == element::f16) {
                data = std::make_shared<opset8::Convert>(data, element::f32);
                ov::mark_as_decompression(data);
            }
            auto input_low = opset8::Constant::create(element::f32, Shape{}, {param.in_low});
            auto input_high = opset8::Constant::create(element::f32, Shape{}, {param.in_high});
            auto output_low = opset8::Constant::create(element::f32, Shape{}, {param.out_low});
            auto output_high = opset8::Constant::create(element::f32, Shape{}, {param.out_high});
            auto fq = std::make_shared<opset8::FakeQuantize>(data,
                                                             input_low,
                                                             input_high,
                                                             output_low,
                                                             output_high,
                                                             param.levels);
            model = std::make_shared<Model>(fq, ParameterVector{});
        }

        manager.register_pass<ov::pass::CompressQuantizeWeights>();

        {
            auto data = opset8::Constant::create(param.expected_type, param.shape, param.expected_weights);
            auto convert = std::make_shared<opset8::Convert>(data, element::f32);
            auto scale = opset8::Constant::create(element::f32, Shape{}, {param.scale_val});
            std::shared_ptr<opset8::Multiply> mul;
            if (!param.fuse_zero_point) {
                auto zero_point = opset8::Constant::create(element::f32, Shape{}, {param.zero_point_val});
                auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
                mul = std::make_shared<opset8::Multiply>(sub, scale);
            } else {
                mul = std::make_shared<opset8::Multiply>(convert, scale);
            }
            model_ref = std::make_shared<Model>(mul, ParameterVector{});
        }
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    }
};

#ifdef OPENVINO_ARCH_ARM64
// Ticket: CVS-122397
TEST_P(CompressQuantizeWeightsTests, DISABLED_FusionTest) {}
#else
TEST_P(CompressQuantizeWeightsTests, FusionTest) {}
#endif

static std::vector<CompressQuantizeWeightsParams> params = {
    {Shape{2, 3, 1, 1},
     {-1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f},
     0.0f,
     10.0f,
     -1.0f,
     5.0f,
     3,
     element::i4,
     {-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
     3.0f,
     -0.666667f,
     false},
    {Shape{2, 3, 1, 1},
     {-1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f},
     0.0f,
     10.0f,
     -1.0f,
     4.0f,
     16,
     element::i4,
     {-8.0f, -5.0f, -4.0f, -2.0f, 0.0f, 7.0f},
     0.333333f,
     -5.0f,
     false},
    {Shape{2, 4, 1, 1},
     {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f},
     1.0f,
     9.0f,
     -2.0f,
     6.0f,
     17,
     element::i8,
     {-4.0f, -4.0f, -4.0f, -2.0f, 0.0f, 2.0f, 4.0f, 12.0f},
     0.5f,
     -4.0f,
     true},
    {Shape{2, 4, 1, 1},
     {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f},
     1.0f,
     9.0f,
     -2.0f,
     6.0f,
     256,
     element::i8,
     {-128.0f, -128.0f, -128.0f, -96.0f, -64.0f, -32.0f, 0.0f, 127.0f},
     0.0313725f,
     -64.25f,
     false},
};

static element::TypeVector data_precisions = {element::f32, element::f16};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         CompressQuantizeWeightsTests,
                         ::testing::Combine(::testing::ValuesIn(params), ::testing::ValuesIn(data_precisions)));

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

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }
    {
        auto data = opset8::Constant::create(element::i8, Shape{2, 4, 1, 1}, {-128, -128, -128, -96, -64, -32, 0, 127});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {10.0 / 255});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {2 - 255.0 / 10});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, CompressQuantizeWeightsWithDequantizationSubgraphFP16) {
    {
        auto data = opset8::Constant::create(element::f16, Shape{2, 4, 1, 1}, {-1, 0, 1, 2, 3, 4, 5, 11});
        auto convert_to_f32 = std::make_shared<opset8::Convert>(data, element::f32);
        ov::mark_as_decompression(convert_to_f32);
        auto input_low = opset8::Constant::create(element::f32, Shape{}, {1});
        auto input_high = opset8::Constant::create(element::f32, Shape{}, {9});
        auto output_low = opset8::Constant::create(element::f32, Shape{}, {-128});
        auto output_high = opset8::Constant::create(element::f32, Shape{}, {127});
        auto fq =
            std::make_shared<opset8::FakeQuantize>(convert_to_f32, input_low, input_high, output_low, output_high, 256);
        auto convert = std::make_shared<opset8::Convert>(fq, element::i8);
        auto second_convert = std::make_shared<opset8::Convert>(convert, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {10.0 / 255});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {2 - 255.0 / 10});
        auto sub = std::make_shared<opset8::Subtract>(second_convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }
    {
        auto data = opset8::Constant::create(element::i8, Shape{2, 4, 1, 1}, {-128, -128, -128, -96, -64, -32, 0, 127});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {10.0 / 255});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {2 - 255.0 / 10});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, CompressQuantizeWeightsWithZeroPointEliminated) {
    {
        auto data = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.144816, 0.0858578, 0.110928});
        auto input_low = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.402659, -0.383148, -0.34054});
        auto input_high = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.399513, 0.380155, 0.33788});
        auto output_low = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.402659, -0.383148, -0.34054});
        auto output_high = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.399513, 0.380155, 0.33788});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }

    {
        auto data = opset8::Constant::create(element::i8, Shape{3, 1, 1, 1}, {-46, 29, 42});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.00314577, 0.00299335, 0.00266047});
        auto mul = std::make_shared<opset8::Multiply>(convert, scale);
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, CompressQuantizeWeightsWithZeroPointEliminatedZeroScale) {
    {
        auto data = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.144816, 0.0858578, 0.110928});
        auto input_low = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.402659, -0.383148, -0.34054});
        auto input_high = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.399513, 0.380155, 0.33788});
        auto output_low = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.402659, 0.0, -0.34054});
        auto output_high = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.399513, 0.0, 0.33788});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }

    {
        auto data = opset8::Constant::create(element::i8, Shape{3, 1, 1, 1}, {-46, 29, 42});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.00314577, 0.0, 0.00266047});
        auto mul = std::make_shared<opset8::Multiply>(convert, scale);
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, CompressQuantizeWeightsWithZeroPointEliminatedFP16) {
    {
        auto data = opset8::Constant::create(element::f16, Shape{3, 1, 1, 1}, {0.2, 1.2, 1.2});
        auto input_low =
            opset8::Constant::create(element::f16, Shape{3, 1, 1, 1}, {0.59033203125, 1.4833984375, 1.2900390625});
        auto input_high =
            opset8::Constant::create(element::f16, Shape{3, 1, 1, 1}, {-0.59033203125, -1.4833984375, -1.2900390625});
        auto output_low =
            opset8::Constant::create(element::f16, Shape{3, 1, 1, 1}, {0.295166015625, 0.74169921875, 0.64501953125});
        auto output_high = opset8::Constant::create(element::f16,
                                                    Shape{3, 1, 1, 1},
                                                    {-0.295166015625, -0.74169921875, -0.64501953125});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 255);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }

    {
        auto data = opset8::Constant::create(element::i8, Shape{3, 1, 1, 1}, {-43, -103, -118});
        auto convert = std::make_shared<opset8::Convert>(data, element::f16);
        auto scale = opset8::Constant::create(element::f16, Shape{3, 1, 1, 1}, {-0.002325, -0.00584, -0.005077});
        auto mul = std::make_shared<opset8::Multiply>(convert, scale);
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, NegativeCompressQuantizeWeights) {
    {
        auto data = opset8::Constant::create(element::f32, Shape{2, 4, 1, 1}, {-1, 0, 1, 2, 3, 4, 5, 11});
        auto input_low = opset8::Constant::create(element::f32, Shape{}, {1});
        auto input_high = opset8::Constant::create(element::f32, Shape{}, {9});
        auto output_low = opset8::Constant::create(element::f32, Shape{}, {-2});
        auto output_high = opset8::Constant::create(element::f32, Shape{}, {6});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }
    {
        auto data = opset8::Constant::create(element::i8, Shape{2, 4, 1, 1}, {-128, -128, -128, -96, -64, -32, 0, 127});
        auto convert = std::make_shared<opset8::Convert>(data, element::f32);
        auto scale = opset8::Constant::create(element::f32, Shape{}, {0.0313725});
        auto zero_point = opset8::Constant::create(element::f32, Shape{}, {-64.25});
        auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
        auto mul = std::make_shared<opset8::Multiply>(sub, scale);
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});
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
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});

    manager.register_pass<ov::pass::CompressQuantizeWeights>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
