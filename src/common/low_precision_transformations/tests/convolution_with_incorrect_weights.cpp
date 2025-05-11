// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/convolution.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/fold_fake_quantize.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <sstream>
#include <string>
#include "transformations/init_node_info.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/convolution.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
class ConvolutionWithIncorrectWeightsTestValues {
public:
    class Actual {
    public:
        ov::builder::subgraph::DequantizationOperations dequantization;
        ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class Expected {
    public:
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type weightsPrecision;
        std::vector<float> weightsValues;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ov::element::Type inputPrecision;
    ov::Shape inputShape;
    TestTransformationParams params;
    bool isCorrect;
    Actual actual;
    Expected expected;
};

class ConvolutionWithIncorrectWeightsTransformation
    : public LayerTransformation,
      public testing::WithParamInterface<ConvolutionWithIncorrectWeightsTestValues> {
public:
    void SetUp() override {
        const ConvolutionWithIncorrectWeightsTestValues testValues = GetParam();

        actualFunction = ov::builder::subgraph::ConvolutionFunction::getOriginalWithIncorrectWeights(
            testValues.inputShape,
            testValues.inputPrecision,
            testValues.actual.fakeQuantizeOnWeights,
            testValues.actual.dequantization,
            testValues.isCorrect);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::ConvolutionTransformation, ov::opset1::Convolution>(
            testValues.params);
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(
            testValues.params);
        transform.transform(actualFunction);

        ov::pass::Manager cleanupManager;
        cleanupManager.register_pass<ov::pass::low_precision::FoldFakeQuantizeTransformation>();
        cleanupManager.register_pass<ov::pass::ConstantFolding>();
        cleanupManager.run_passes(actualFunction);

        referenceFunction = ov::builder::subgraph::ConvolutionFunction::getReferenceWithIncorrectWeights(
            testValues.inputShape,
            testValues.inputPrecision,
            testValues.expected.dequantizationBefore,
            testValues.expected.weightsPrecision,
            testValues.expected.weightsValues,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionWithIncorrectWeightsTestValues> obj) {
        const ConvolutionWithIncorrectWeightsTestValues testValues = obj.param;

        std::ostringstream result;
        result << toString(testValues.params) << (testValues.isCorrect ? "_correct_weights" : "_incorrect_weights");
        return result.str();
    }
};

TEST_P(ConvolutionWithIncorrectWeightsTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ConvolutionWithIncorrectWeightsTestValues> testValues = {
    // incorrect weights
    {
        ov::element::u8,
        ov::Shape({1, 3, 224, 224}),
        LayerTransformation::createParamsU8I8(),
        false,
        {
            {ov::element::f32, {}, {0.1f}},
            {255ul, ov::Shape{1, 1, 1, 1}, {0.f}, {254.f}, {-127.f}, {127.f}},
        },
        {{ov::element::f32, {}, {0.1f}}, ov::element::f32, {-129.f}, {}},
    },
    // correct weights
    {
        ov::element::u8,
        ov::Shape({1, 3, 224, 224}),
        LayerTransformation::createParamsU8I8(),
        true,
        {
            {ov::element::f32, {}, {0.1f}},
            {255ul, ov::Shape{1, 1, 1, 1}, {0.f}, {254.f}, {-127.f}, {127.f}},
        },
        {
            {},
            ov::element::i8,
            {-126.f},
            {{}, {}, {{0.1f}, ov::element::f32, {}}},
        },
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         ConvolutionWithIncorrectWeightsTransformation,
                         ::testing::ValuesIn(testValues),
                         ConvolutionWithIncorrectWeightsTransformation::getTestCaseName);

}  // namespace
