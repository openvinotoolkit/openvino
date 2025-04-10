// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"
#include <memory>
#include <sstream>
#include <string>
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class PullReshapeThroughDequantizationTestValues {
public:
    class Values {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        ov::builder::subgraph::Constant weights;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ov::builder::subgraph::Reshape reshape1;
        ov::builder::subgraph::DequantizationOperations::Multiply multiply;
        ov::builder::subgraph::Transpose transpose;
        ov::builder::subgraph::Reshape reshape2;
        ov::element::Type precisionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Values actual;
    Values expected;
    std::string operation;
};

typedef std::tuple<ov::Shape,
                   std::pair<ov::Shape, ov::Shape>,
                   ov::Shape,  // multiplyShapes
                   PullReshapeThroughDequantizationTestValues>
    PullReshapeThroughDequantizationParams;

class PullReshapeThroughDequantizationTransformation
    : public LayerTransformation,
      public testing::WithParamInterface<PullReshapeThroughDequantizationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto dequantizationElementwiseShape = std::get<1>(GetParam());
        const auto multiplyShape = std::get<2>(GetParam());
        auto testValues = std::get<3>(GetParam());
        const auto operation = testValues.operation.empty() ? "GroupConvolution" : testValues.operation;

        // to prevent test cases increasing let's parameterize test by dequantization shape and
        // initialize values here
        testValues.actual.dequantizationOnWeights.subtract.constantShape = dequantizationElementwiseShape.first;
        testValues.actual.dequantizationOnWeights.multiply.constantShape = dequantizationElementwiseShape.first;
        testValues.actual.multiply.constantShape = multiplyShape;
        testValues.expected.dequantizationOnWeights.subtract.constantShape = dequantizationElementwiseShape.second;
        testValues.expected.dequantizationOnWeights.multiply.constantShape = dequantizationElementwiseShape.second;

        actualFunction = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
            testValues.actual.precisionBeforeDequantization,
            inputShape,
            {},
            {},
            testValues.actual.dequantizationOnActivations,
            testValues.actual.weights,
            {},
            {},
            testValues.actual.dequantizationOnWeights,
            testValues.actual.reshape1,
            testValues.actual.multiply,
            testValues.actual.transpose,
            testValues.actual.reshape2,
            testValues.actual.dequantizationAfter,
            operation);

        ov::pass::Manager manager;
        auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
        const std::vector<ov::element::Type> supportedTypes = {ov::element::i8, ov::element::u8};
        decomp->add_matcher<ov::pass::low_precision::PullReshapeThroughDequantization>(supportedTypes);
        decomp->add_matcher<ov::pass::low_precision::PullTransposeThroughDequantization>(supportedTypes);
        decomp->add_matcher<ov::pass::LinOpSequenceFusion>();
        manager.run_passes(actualFunction);

        referenceFunction = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
            testValues.actual.precisionBeforeDequantization,
            inputShape,
            {},
            {},
            testValues.expected.dequantizationOnActivations,
            testValues.expected.weights,
            {},
            {},
            testValues.expected.dequantizationOnWeights,
            testValues.expected.reshape1,
            testValues.expected.multiply,
            testValues.expected.transpose,
            testValues.expected.reshape2,
            testValues.expected.dequantizationAfter,
            operation);
    }

    static std::string getTestCaseName(testing::TestParamInfo<PullReshapeThroughDequantizationParams> obj) {
        const auto inputShape = std::get<0>(obj.param);
        const auto dequantizationElementwiseShape = std::get<1>(obj.param);
        const auto multiplyShape = std::get<2>(obj.param);
        const PullReshapeThroughDequantizationTestValues testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" << inputShape << "_" << dequantizationElementwiseShape.first << "_"
               << dequantizationElementwiseShape.second << "_" << multiplyShape << "_"
               << testValues.actual.precisionBeforeDequantization << "_"
               << testValues.actual.dequantizationOnActivations << "_"
               << "_weights_" << testValues.actual.weights.outPrecision << "_"
               << "{ " << testValues.actual.weights.values[0] << " }_" << testValues.actual.dequantizationOnWeights;
        return result.str();
    }
};

TEST_P(PullReshapeThroughDequantizationTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

// clang-format off

const std::vector<ov::Shape> inputShapes = {
    ov::Shape({1, 960, 7, 7}),
    ov::Shape({4, 960, 7, 7})
};

const std::vector<std::pair<ov::Shape, ov::Shape>> dequantizationOnWeightElementwiseConstantShapes = {
    {ov::Shape({1, 960}), ov::Shape({960, 1, 1, 1})},
    {ov::Shape({9, 960}), ov::Shape({960, 1, 3, 3})}
};

const std::vector<ov::Shape> multiplyShapes = {ov::Shape({1, 1, 960, 1})};

const std::vector<PullReshapeThroughDequantizationTestValues> testValues = {
    // Subtract with Constant (no Convert)
    // Actual:
    //                        Constant
    //                         |I8
    //                         |
    //                        Convert  Constant
    //                         |FP32   /FP32
    //                         |      /
    //                        Subtract Constant
    //                         |FP32   /FP32
    //                         |      /
    //                        Multiply Constant
    //                         |FP32   /FP32
    //                         |      /
    // Parameter   Constant   Reshape  Constant
    //  |U8         |U8        |FP32   /FP32
    //  |           |          |      /
    // Convert    Convert     Multiply  Constant
    //   \FP32    /FP32        |FP32   /FP32
    //    \      /             |      /
    //    Subtract  Constant  Transpose  Constant
    //      \FP32   /FP32      |FP32   /I64
    //       \     /           |      /
    //       Multiply         Reshape
    //         \FP32         /FP32
    //          \           /
    //        GroupConvolution
    //
    //
    // Transformed:
    //
    //                      Constant
    //                       |I8
    //                       |
    // Parameter Constant   Convert  Constant
    //  |U8       |U8        |FP32   /FP32
    //  |         |          |      /
    // Convert   Convert    Subtract  Constant
    //  \FP32   /FP32        |FP32   /FP32
    //   \     /             |      /
    //   Subtract  Constant  Multiply Constant
    //     \FP32   /FP32     |FP32   /I64
    //      \     /          |      /
    //       Multiply       Reshape
    //         \FP32        /FP32
    //          \          /
    //       GroupConvolution  Constant
    //            \FP32       /FP32
    //             \         /
    //               Multiply
    //
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8, {9, 960}},
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {/* from parameter */}, false },
                { {0.03f}, element::f32, {/* from parameter */}, false }
            },
            { {3, 3, 960, 1} },
            { {2}, element::f32, {/* from parameter: multiplyShapes */}, false },
            { {2, 3, 0, 1} },
            { {960, 1, 1, 3, 3} },
            ov::element::f32,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8, {960, 1, 3, 3}},
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {/* from parameter */}, false },
                { {0.06f}, element::f32, {/* from parameter */}, false }
            },
            {},
            {},
            {},
            {{960, 1, 1, 3, 3}},
            ov::element::f32,
            {}
        }
    },

    // Subtract with Convert + Constant
    // Actual:
    //                        Constant Constant
    //                         |I8      |I8
    //                         |        |
    //                        Convert  Convert
    //                         |FP32   /FP32
    //                         |      /
    //                        Subtract Constant
    //                         |FP32   /FP32
    //                         |      /
    //                        Multiply Constant
    //                         |FP32   /FP32
    //                         |      /
    // Parameter   Constant   Reshape  Constant
    //  |U8         |U8        |FP32   /FP32
    //  |           |          |      /
    // Convert    Convert     Multiply  Constant
    //   \FP32    /FP32        |FP32   /FP32
    //    \      /             |      /
    //    Subtract  Constant  Transpose  Constant
    //      \FP32   /FP32      |FP32   /I64
    //       \     /           |      /
    //       Multiply         Reshape
    //         \FP32         /FP32
    //          \           /
    //        GroupConvolution
    //
    //
    // Transformed:
    //
    //                      Constant
    //                       |I8
    //                       |
    // Parameter Constant   Convert  Constant
    //  |U8       |U8        |FP32   /FP32
    //  |         |          |      /
    // Convert   Convert    Subtract  Constant
    //  \FP32   /FP32        |FP32   /FP32
    //   \     /             |      /
    //   Subtract  Constant  Multiply Constant
    //     \FP32   /FP32     |FP32   /I64
    //      \     /          |      /
    //       Multiply       Reshape
    //         \FP32        /FP32
    //          \          /
    //       GroupConvolution  Constant
    //            \FP32       /FP32
    //             \         /
    //               Multiply
    //
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8, {9, 960}},
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {/* from parameter */}, false, 1ul, element::i8, true },
                { {0.03f}, element::f32, {/* from parameter */}, false }
            },
            { {3, 3, 960, 1} },
            { {2}, element::f32, {/* from parameter: multiplyShapes */}, false },
            { {2, 3, 0, 1} },
            { {960, 1, 1, 3, 3} },
            ov::element::f32,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8, {960, 1, 3, 3}},
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {/* from parameter */}, false, 1ul, element::i8, true },
                { {0.06f}, element::f32, {/* from parameter */}, false }
            },
            {},
            {},
            {},
            {{960, 1, 1, 3, 3}},
            ov::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         PullReshapeThroughDequantizationTransformation,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(dequantizationOnWeightElementwiseConstantShapes),
                                            ::testing::ValuesIn(multiplyShapes),
                                            ::testing::ValuesIn(testValues)),
                         PullReshapeThroughDequantizationTransformation::getTestCaseName);

namespace reshapeAsTranspose {
const PullReshapeThroughDequantizationTestValues reshapeAsTransposeTestCase = {
    LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
    // ActualValues
    {
        element::u8,
        {{element::f32, false}, {}, {{0.02f}, element::f32, {}, false}},
        {std::vector<float>{2.f}, element::i8, {24, 1, 1, 32}},
        {{element::f32, false}, {}, {{0.03f}, element::f32, {/* from parameter */}, false}},
        {},
        {},
        {{}},
        {{24, 32, 1, 1}},
        element::f32,
        {}
    },
    // ExpectedValues
    {
        element::u8,
        {{element::f32, false}, {}, {{0.02f}, element::f32, {}, false}},
        {std::vector<float>{2.f}, element::i8, {24, 32, 1, 1}},
        {{element::f32, false}, {}, {{0.03f}, element::f32, {/* from parameter */}, false}},
        {},
        {},
        {},
        {},
        element::f32,
        {}
    },
    "Convolution"
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         PullReshapeThroughDequantizationTransformation,
                         ::testing::Combine(::testing::Values(Shape{1, 32, 150, 150}),
                                            ::testing::Values(std::pair<Shape, Shape>{{24, 1, 1, 1}, {24, 1, 1, 1}}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(reshapeAsTransposeTestCase)),
                         PullReshapeThroughDequantizationTransformation::getTestCaseName);
// clang-format on
}  // namespace reshapeAsTranspose
