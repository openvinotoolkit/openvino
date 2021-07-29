// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/pull_reshape_through_dequantization.hpp>
#include <low_precision/pull_transpose_through_dequantization.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/fake_quantize_and_convolution_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class PullReshapeThroughDequantizationTestValues {
public:
    class Values {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        ngraph::builder::subgraph::Constant weights;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ngraph::builder::subgraph::Reshape reshape1;
        ngraph::builder::subgraph::DequantizationOperations::Multiply multiply;
        ngraph::builder::subgraph::Transpose transpose;
        ngraph::builder::subgraph::Reshape reshape2;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Values actual;
    Values expected;
};

typedef std::tuple<
    ngraph::Shape,
    std::pair<ngraph::Shape, ngraph::Shape>,
    ngraph::Shape, // multiplyShapes
    PullReshapeThroughDequantizationTestValues> PullReshapeThroughDequantizationParams;

class PullReshapeThroughDequantizationTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<PullReshapeThroughDequantizationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto dequantizationElementwiseShape = std::get<1>(GetParam());
        const auto multiplyShape = std::get<2>(GetParam());
        auto testValues = std::get<3>(GetParam());

        // to prevent test cases increasing let's parameterize test by dequantization shape and
        // initialize values here
        testValues.actual.dequantizationOnWeights.subtract.constantShape = dequantizationElementwiseShape.first;
        testValues.actual.dequantizationOnWeights.multiply.constantShape = dequantizationElementwiseShape.first;
        testValues.actual.multiply.constantShape = multiplyShape;
        testValues.expected.dequantizationOnWeights.subtract.constantShape = dequantizationElementwiseShape.second;
        testValues.expected.dequantizationOnWeights.multiply.constantShape = dequantizationElementwiseShape.second;

        actualFunction = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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
            "GroupConvolution");

        ngraph::pass::Manager manager;
        auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
        const std::vector<ngraph::element::Type> supportedTypes = { ngraph::element::i8, ngraph::element::u8 };
        decomp->add_matcher<ngraph::pass::low_precision::PullReshapeThroughDequantization>(supportedTypes);
        decomp->add_matcher<ngraph::pass::low_precision::PullTransposeThroughDequantization>(supportedTypes);
        decomp->add_matcher<ngraph::pass::LinOpSequenceFusion>();
        manager.run_passes(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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
            "GroupConvolution");
    }

    static std::string getTestCaseName(testing::TestParamInfo<PullReshapeThroughDequantizationParams> obj) {
        const auto inputShape = std::get<0>(obj.param);
        const auto dequantizationElementwiseShape = std::get<1>(obj.param);
        const auto multiplyShape = std::get<2>(obj.param);
        const PullReshapeThroughDequantizationTestValues testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
            inputShape << "_" <<
            dequantizationElementwiseShape.first << "_" <<
            dequantizationElementwiseShape.second << "_" <<
            multiplyShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantizationOnActivations << "_" << "_weights_" <<
            testValues.actual.weights.outPrecision << "_" << "{ " <<
            testValues.actual.weights.values[0] << " }_" <<
            testValues.actual.dequantizationOnWeights;
        return result.str();
    }
};

TEST_P(PullReshapeThroughDequantizationTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> inputShapes = {
    ngraph::Shape({ 1, 960, 7, 7 }),
    ngraph::Shape({ 4, 960, 7, 7 })
};

const std::vector<std::pair<ngraph::Shape, ngraph::Shape>> dequantizationOnWeightElementwiseConstantShapes = {
    { ngraph::Shape({1, 960}), ngraph::Shape({960, 1, 1, 1}) },
    { ngraph::Shape({9, 960}), ngraph::Shape({960, 1, 3, 3}) }
};

const std::vector<ngraph::Shape> multiplyShapes = {
    ngraph::Shape({1, 1, 960, 1})
};

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
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8, {9, 960}},
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {/* from parameter */}, false },
                { {0.03f}, element::f32, {/* from parameter */}, false }
            },
            { {3, 3, 960, 1} },
            { {2}, element::f32, {/* from parameter: multiplyShapes */}, false },
            { {2, 3, 0, 1} },
            { {960, 1, 1, 3, 3} },
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8, {960, 1, 3, 3}},
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {/* from parameter */}, false },
                { {0.06f}, element::f32, {/* from parameter */}, false }
            },
            {},
            {},
            {},
            {{960, 1, 1, 3, 3}},
            ngraph::element::f32,
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
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8, {9, 960}},
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {/* from parameter */}, false, 1ul, element::i8, true },
                { {0.03f}, element::f32, {/* from parameter */}, false }
            },
            { {3, 3, 960, 1} },
            { {2}, element::f32, {/* from parameter: multiplyShapes */}, false },
            { {2, 3, 0, 1} },
            { {960, 1, 1, 3, 3} },
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8, {960, 1, 3, 3}},
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {/* from parameter */}, false, 1ul, element::i8, true },
                { {0.06f}, element::f32, {/* from parameter */}, false }
            },
            {},
            {},
            {},
            {{960, 1, 1, 3, 3}},
            ngraph::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PullReshapeThroughDequantizationTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(dequantizationOnWeightElementwiseConstantShapes),
        ::testing::ValuesIn(multiplyShapes),
        ::testing::ValuesIn(testValues)),
    PullReshapeThroughDequantizationTransformation::getTestCaseName);
