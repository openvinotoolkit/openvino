// Copyright (C) 2018-2025 Intel Corporation
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
    {Shape{2, 3, 1, 1},
     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f},
     0.0f,
     10.0f,
     0.0f,
     5.0f,
     3,
     element::u4,
     {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f},
     2.5f,
     -0.5f,
     true},
    {Shape{2, 3, 1, 1},
     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f},
     0.0f,
     10.0f,
     1.0f,
     4.0f,
     16,
     element::u4,
     {2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 15.0f},
     0.2f,
     -5.0f,
     false},
    {Shape{2, 4, 1, 1},
     {1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f},
     1.0f,
     9.0f,
     2.0f,
     6.0f,
     17,
     element::u8,
     {8.0f, 8.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 24.0f},
     0.25f,
     -4.0f,
     true},
    {Shape{2, 4, 1, 1},
     {1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f},
     1.0f,
     9.0f,
     2.0f,
     6.0f,
     256,
     element::u8,
     {0, 0, 0, 32.0f, 64.0f, 96.0f, 128.0f, 255.0f},
     0.0156863f,
     -127.5f,
     false},
};

static element::TypeVector data_precisions = {element::f32, element::f16};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         CompressQuantizeWeightsTests,
                         ::testing::Combine(::testing::ValuesIn(params), ::testing::ValuesIn(data_precisions)));

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 122666
TEST_F(TransformationTestsF, DISABLED_CompressQuantizeWeightsWithDequantizationSubgraph) {
#else
TEST_F(TransformationTestsF, CompressQuantizeWeightsWithDequantizationSubgraph) {
#endif
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

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 122666
TEST_F(TransformationTestsF, DISABLED_CompressQuantizeWeightsWithDequantizationSubgraphFP16) {
#else
TEST_F(TransformationTestsF, CompressQuantizeWeightsWithDequantizationSubgraphFP16) {
#endif
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

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 122666
TEST_F(TransformationTestsF, DISABLED_CompressQuantizeWeightsWithZeroPointEliminated) {
#else
TEST_F(TransformationTestsF, CompressQuantizeWeightsWithZeroPointEliminated) {
#endif
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

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 122666
TEST_F(TransformationTestsF, DISABLED_CompressQuantizeWeightsWithZeroPointEliminatedZeroScale) {
#else
TEST_F(TransformationTestsF, CompressQuantizeWeightsWithZeroPointEliminatedZeroScale) {
#endif
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

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 122666
TEST_F(TransformationTestsF, DISABLED_CompressQuantizeWeightsWithZeroPointEliminatedFP16) {
#else
TEST_F(TransformationTestsF, CompressQuantizeWeightsWithZeroPointEliminatedFP16) {
#endif
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

TEST_F(TransformationTestsF, CompressQuantizeWeightsWithZeroPointEliminatedBF16) {
    {
        auto data = opset8::Constant::create(element::bf16, Shape{3, 1, 1, 1}, {0.2, 1.2, 1.2});
        auto input_low = opset8::Constant::create(element::bf16, Shape{3, 1, 1, 1}, {0.60, 1.45, 1.30});
        auto input_high = opset8::Constant::create(element::bf16, Shape{3, 1, 1, 1}, {-0.60, -1.45, -1.30});
        auto output_low = opset8::Constant::create(element::bf16, Shape{3, 1, 1, 1}, {0.30, 0.75, 0.65});
        auto output_high = opset8::Constant::create(element::bf16, Shape{3, 1, 1, 1}, {-0.30, -0.75, -0.65});
        auto fq = std::make_shared<opset8::FakeQuantize>(data, input_low, input_high, output_low, output_high, 255);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }

    {
        auto data = opset8::Constant::create(element::i8, Shape{3, 1, 1, 1}, {-42, -105, -118});
        auto convert = std::make_shared<opset8::Convert>(data, element::bf16);
        auto scale = opset8::Constant::create(element::bf16, Shape{3, 1, 1, 1}, {-0.002325, -0.00592, -0.00509});
        auto mul = std::make_shared<opset8::Multiply>(convert, scale);
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);

    m_abs_threshold = 4e-2f;
    m_rel_threshold = 7e-2f;
}

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 122666
TEST_F(TransformationTestsF, DISABLED_NegativeCompressQuantizeWeights) {
#else
TEST_F(TransformationTestsF, NegativeCompressQuantizeWeights) {
#endif
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

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 122666
TEST_F(TransformationTestsF, DISABLED_NegativeCompressQuantizeWeightsNonConstantInput) {
#else
TEST_F(TransformationTestsF, NegativeCompressQuantizeWeightsNonConstantInput) {
#endif
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

using CompressWeightsWithFakeConvertParams = std::tuple<bool,          // zero_point_absent
                                                        std::string>;  // float8 type

class CompressWeightsNoZeroPoint : public TransformationTestsF,
                                   public testing::WithParamInterface<CompressWeightsWithFakeConvertParams> {};

TEST_P(CompressWeightsNoZeroPoint, FakeConvert) {
    const auto& param = GetParam();
    bool zero_point_absent = std::get<0>(param);
    std::string destination_type = std::get<1>(param);

    {
        auto weights = op::v0::Constant::create(element::f32,
                                                Shape{3, 1, 2, 2},
                                                {-0.01448f,
                                                 -0.02314f,
                                                 -0.02244f,
                                                 -0.00090f,
                                                 0.024261f,
                                                 0.031921f,
                                                 0.034088f,
                                                 -0.0497f,
                                                 -0.0588f,
                                                 -0.04541f,
                                                 -0.01281f,
                                                 0.009109f});
        auto scale = op::v0::Constant::create(element::f32, Shape{3, 1, 1, 1}, {54.50976f});
        std::shared_ptr<op::v13::FakeConvert> fake_convert;
        if (zero_point_absent) {
            fake_convert = std::make_shared<op::v13::FakeConvert>(weights, scale, destination_type);
        } else {
            auto shift = op::v0::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.0f});
            fake_convert = std::make_shared<op::v13::FakeConvert>(weights, scale, shift, destination_type);
        }
        model = std::make_shared<Model>(fake_convert, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }

    {
        std::vector<float> weights_data =
            destination_type == "f8e4m3"
                ? std::vector<float>{-0.8125f,
                                     -1.25f,
                                     -1.25f,
                                     -0.0507812f,
                                     1.375f,
                                     1.75f,
                                     1.875f,
                                     -2.75f,
                                     -3.25f,
                                     -2.5f,
                                     -0.6875f,
                                     0.5f}
                :

                std::vector<
                    float>{-0.75f, -1.25f, -1.25f, -0.046875f, 1.25f, 1.75f, 1.75f, -2.5f, -3.0f, -2.5f, -0.75f, 0.5f};
        auto weights =
            std::make_shared<op::v0::Constant>(element::Type(destination_type), Shape{3, 1, 2, 2}, weights_data);
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        auto scale = op::v0::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.01834533f});
        auto multiply = std::make_shared<op::v1::Multiply>(convert, scale);
        model_ref = std::make_shared<Model>(multiply, ParameterVector{});
    }

    m_abs_threshold = 1e-6f;
    m_rel_threshold = 1e-6f;

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(CompressQuantizeWeights,
                         CompressWeightsNoZeroPoint,
                         testing::Combine(testing::Values(false, true), testing::Values("f8e4m3", "f8e5m2")));

class CompressWeightsWithZeroPoint : public TransformationTestsF, public testing::WithParamInterface<std::string> {};

TEST_P(CompressWeightsWithZeroPoint, FakeConvert) {
    const auto& destination_type = GetParam();

    {
        auto weights = op::v0::Constant::create(element::f32,
                                                Shape{3, 1, 2, 2},
                                                {-0.01448f,
                                                 -0.02314f,
                                                 -0.02244f,
                                                 -0.00090f,
                                                 0.024261f,
                                                 0.031921f,
                                                 0.034088f,
                                                 -0.0497f,
                                                 -0.0588f,
                                                 -0.04541f,
                                                 -0.01281f,
                                                 0.009109f});
        auto scale = op::v0::Constant::create(element::f32, Shape{3, 1, 1, 1}, {54.50976f});
        auto shift = op::v0::Constant::create(element::f32, Shape{3, 1, 1, 1}, {0.7f, -0.0304f, -0.012f});
        auto fake_convert = std::make_shared<op::v13::FakeConvert>(weights, scale, shift, destination_type);
        model = std::make_shared<Model>(fake_convert, ParameterVector{});

        manager.register_pass<ov::pass::CompressQuantizeWeights>();
    }

    {
        std::vector<float> weights_data =
            destination_type == "f8e4m3"
                ? std::vector<float>{-1.5f,
                                     -2.0f,
                                     -1.875f,
                                     -0.75f,
                                     1.375f,
                                     1.75f,
                                     1.875f,
                                     -2.75f,
                                     -3.25f,
                                     -2.5f,
                                     -0.6875f,
                                     0.5f}
                : std::vector<
                      float>{-1.5f, -2.0f, -2.0f, -0.75f, 1.25f, 1.75f, 2.0f, -2.5f, -3.0f, -2.5f, -0.625f, 0.5f};

        auto weights =
            std::make_shared<op::v0::Constant>(element::Type(destination_type), Shape{3, 1, 2, 2}, weights_data);
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        auto shift = op::v0::Constant::create(element::f32, Shape{3, 1, 1, 1}, {-0.7f, 0.0304f, 0.012f});
        auto subtract = std::make_shared<op::v1::Subtract>(convert, shift);
        auto scale = op::v0::Constant::create(element::f32, Shape{3, 1, 1, 1}, {1.0f / 54.50976f});
        auto multiply = std::make_shared<op::v1::Multiply>(subtract, scale);
        model_ref = std::make_shared<Model>(multiply, ParameterVector{});
    }

    m_abs_threshold = 1e-6f;
    m_rel_threshold = 1e-6f;

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(CompressQuantizeWeights, CompressWeightsWithZeroPoint, testing::Values("f8e4m3", "f8e5m2"));
