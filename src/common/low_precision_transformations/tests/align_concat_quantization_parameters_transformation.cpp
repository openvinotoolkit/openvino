// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/avg_pool.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/max_pool.hpp"
#include <memory>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/align_concat_quantization_parameters.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov::pass;

class AlignConcatQuantizationParametersTransformationTestValues {
public:
public:
    class Actual {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type preicsionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<ov::element::Type,
                   ov::Shape,
                   bool,         // additional FakeQuantize After
                   std::string,  // additional layer before FQ
                   AlignConcatQuantizationParametersTransformationTestValues>
    AlignConcatQuantizationParametersTransformationParams;

class AlignConcatQuantizationParametersTransformation
    : public LayerTransformation,
      public testing::WithParamInterface<AlignConcatQuantizationParametersTransformationParams> {
public:
    void SetUp() override {
        ov::element::Type precision;
        ov::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AlignConcatQuantizationParametersTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = GetParam();

        actualFunction = ov::builder::subgraph::AlignConcatQuantizationParametersFunction::getOriginal(
            precision,
            testValues.actual.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.actual.dequantization);

        auto supportedPrecisions = std::vector<ov::pass::low_precision::PrecisionsRestriction>(
            {ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::Convolution>(
                {{{0}, {ov::element::u8}}, {{1}, {ov::element::i8}}})});

        auto perTensorQuantization = std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>({
            ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::op::v1::Convolution>({0}),
        });

        SimpleLowPrecisionTransformer transform(supportedPrecisions, perTensorQuantization);
        transform.add<ov::pass::low_precision::AvgPoolTransformation, ov::op::v1::AvgPool>(testValues.params);
        transform.add<ov::pass::low_precision::ConcatTransformation, ov::op::v0::Concat>(testValues.params);
        transform.add<ov::pass::low_precision::ConvolutionTransformation, ov::op::v1::Convolution>(
            testValues.params);
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(
            testValues.params);
        transform.add<ov::pass::low_precision::MaxPoolTransformation, ov::op::v1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::AlignConcatQuantizationParametersFunction::getReference(
            precision,
            testValues.expected.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.expected.dequantizationBefore,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(
        testing::TestParamInfo<AlignConcatQuantizationParametersTransformationParams> obj) {
        ov::element::Type precision;
        ov::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AlignConcatQuantizationParametersTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = obj.param;

        std::ostringstream result;
        result << precision << "_"
               << LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision,
                                                               shape,
                                                               testValues.params)
               << "_" << testValues.actual.dequantization << "_" << testValues.expected.dequantizationBefore << "_"
               << testValues.expected.preicsionAfterOperation << "_" << testValues.expected.dequantizationAfter << "_"
               << (addFakeQuantize ? "_FQ_after_" : "_") << additionalLayer;
        return result.str();
    }
};

TEST_P(AlignConcatQuantizationParametersTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {ov::element::f32};

const std::vector<std::string> additionalLayer = {
    "maxpool"  // any transparent layer
};

const std::vector<bool> addFQ = {false};

const std::vector<ov::Shape> shapes = {{1, 3, 9, 9}, {4, 3, 9, 9}};

const std::vector<AlignConcatQuantizationParametersTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {LayerTransformation::createParamsU8I8(),
     {ov::element::f32, {{ov::element::f32}, {128.f}, {0.02f}}},
     {ov::element::f32,
      {{}, {std::vector<float>(6, 128.f), element::f32, {1, 6, 1, 1}}, {}},
      ov::element::f32,
      {{}, {}, {{0.0001f}, element::f32, {}}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         AlignConcatQuantizationParametersTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(addFQ),
                                            ::testing::ValuesIn(additionalLayer),
                                            ::testing::ValuesIn(testValues)),
                         AlignConcatQuantizationParametersTransformation::getTestCaseName);
