// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <low_precision/common/operation_per_tensor_quantization_restriction.hpp>
#include <low_precision/common/operation_precision_restriction.hpp>
#include <low_precision/concat.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/fuse_multiply_to_fake_quantize.hpp>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include <low_precision/rt_info/intervals_alignment_attribute.hpp>
#include <low_precision/rt_info/precision_preserved_attribute.hpp>
#include <low_precision/rt_info/quantization_alignment_attribute.hpp>
#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "layer_transformation.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/lstm_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

namespace {

class LSTMTransformationActualValues {
public:
    std::vector<ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant> fakeQuantizes;
    std::vector<ngraph::builder::subgraph::DequantizationOperations::Convert> converts;
    std::vector<ngraph::builder::subgraph::DequantizationOperations> dequantizations;
};

inline std::ostream& operator<<(std::ostream& out, const LSTMTransformationActualValues& values) {
    return out << "_" << values.fakeQuantizes[0] << "_" << values.converts[0].outPrecision << "_"
               << values.dequantizations[0];
}

class LSTMTransformationResultValues {
public:
    std::vector<ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant> fakeQuantizes;
    std::vector<ngraph::builder::subgraph::DequantizationOperations::Convert> converts;
    std::vector<ngraph::builder::subgraph::DequantizationOperations> dequantizations;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const LSTMTransformationResultValues& values) {
    return out << "_" << values.fakeQuantizes[0] << "_" << values.converts[0].outPrecision << "_"
               << values.dequantizations[0];
}

class LSTMTransformationTestValues {
public:
    LSTMTransformationTestValues() = default;
    LSTMTransformationTestValues(const TestTransformationParams& params,
                                 const bool multiChannels,
                                 const std::int64_t axis,
                                 const LSTMTransformationActualValues& actual,
                                 const LSTMTransformationResultValues& result,
                                 const bool addNotPrecisionPreservedOperation = false,
                                 const bool checkIntervalsAlignmentAttributes = true)
        : params(params),
          multiChannels(multiChannels),
          axis(axis),
          actual(actual),
          result(result),
          addNotPrecisionPreservedOperation(addNotPrecisionPreservedOperation),
          checkIntervalsAlignmentAttributes(checkIntervalsAlignmentAttributes) {}

    TestTransformationParams params;
    bool multiChannels;
    std::int64_t axis;
    LSTMTransformationActualValues actual;
    LSTMTransformationResultValues result;
    // add not precision preserved operation to set output precision for FakeQuantize
    // don't set to 'true' by default to keep test cases with tested operation as output
    bool addNotPrecisionPreservedOperation;
    bool checkIntervalsAlignmentAttributes;
};

inline std::ostream& operator<<(std::ostream& out, const LSTMTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple<ngraph::element::Type, std::vector<ngraph::PartialShape>, LSTMTransformationTestValues>
    LSTMTransformationParams;

class LSTMTransformation : public LayerTransformation, public testing::WithParamInterface<LSTMTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const std::vector<ngraph::PartialShape> shapes = std::get<1>(GetParam());
        LSTMTransformationTestValues testValues = std::get<2>(GetParam());

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        /*if (!testValues.actual.dequantization1.multiply.empty()) {
            testValues.actual.dequantization1.multiply.outPrecision = precision;
        }
        if (!testValues.actual.dequantization2.multiply.empty()) {
            testValues.actual.dequantization2.multiply.outPrecision = precision;
        }*/

        actualFunction = ngraph::builder::subgraph::LSTMFunction::get(precision,
                                                                      shapes,
                                                                      testValues.actual.fakeQuantizes,
                                                                      testValues.actual.converts,
                                                                      testValues.actual.dequantizations,
                                                                      {},
                                                                      ngraph::element::undefined,
                                                                      {});
        ngraph::pass::VisualizeTree("C:\\Users\\ndemasho\\rep\\Visual\\test.actual.dot")
            .run_on_function(actualFunction);
        auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>(
            {ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::AvgPool>(
                {{0, testValues.params.precisionsOnActivations}})});

        auto quantizationRestrictions =
            testValues.multiChannels
                ? std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>()
                : std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>(
                      {ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction::create<
                          ngraph::opset1::AvgPool>()});

        const auto params = TestTransformationParams::toParams(testValues.params);
        SimpleLowPrecisionTransformer transformer(supportedPrecisionsOnActivation, quantizationRestrictions);
        transformer.commonGraphRewrite
            ->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>(params);
        transformer.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::ConcatTransformation>(params);
        transformer.transform(actualFunction);
        ngraph::pass::VisualizeTree("C:\\Users\\ndemasho\\rep\\Visual\\test.transform.dot")
            .run_on_function(actualFunction);
        {
            ngraph::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager
                .register_pass<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager
                .register_pass<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.result.dequantizationAfter.multiply.empty()) {
            testValues.result.dequantizationAfter.multiply.outPrecision = precision;
        }

        if (!testValues.params.updatePrecisions && (precision == ngraph::element::f32) &&
            !testValues.result.dequantizationAfter.convert.empty()) {
            testValues.result.dequantizationAfter.convert = {};
        }

        IntervalsAlignmentSharedValue::Interval interval{-1.28f, 2.55f};

        referenceFunction = ngraph::builder::subgraph::LSTMFunction::get(precision,
                                                                         shapes,
                                                                         testValues.result.fakeQuantizes,
                                                                         testValues.result.converts,
                                                                         testValues.result.dequantizations,
                                                                         {PrecisionPreservedAttribute(true),
                                                                          IntervalsAlignmentAttribute(interval, 256),
                                                                          QuantizationAlignmentAttribute(false)},
                                                                         testValues.result.precisionAfterOperation,
                                                                         testValues.result.dequantizationAfter);
        ngraph::pass::VisualizeTree("C:\\Users\\ndemasho\\rep\\Visual\\test.reference.dot")
            .run_on_function(referenceFunction);
    }

    static std::string getTestCaseName(testing::TestParamInfo<LSTMTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const std::vector<ngraph::PartialShape> shapes = std::get<1>(obj.param);
        const LSTMTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shapes[0], testValues.params) << "_"
               << (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") << "axis_" << testValues.axis
               << "_" << testValues.actual << "_" << testValues.result << "_";
        return result.str();
    }
};

TEST_P(LSTMTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";

    LSTMTransformationTestValues testValues = std::get<2>(GetParam());
    const auto actualFakeQuantizes = LayerTransformation::get<opset1::FakeQuantize>(actualFunction);
    if (testValues.axis == 1) {
        ASSERT_TRUE(checkIfOutputAttributesSharedValuesAreTheSame<PrecisionsAttribute>(actualFakeQuantizes))
            << "PrecisionsAttribute are not the same";

        if (testValues.checkIntervalsAlignmentAttributes) {
            auto operations = LayerTransformation::get<opset1::Concat>(actualFunction);
            operations.insert(operations.end(), actualFakeQuantizes.begin(), actualFakeQuantizes.end());
            ASSERT_TRUE(checkIfAttributesSharedValuesAreTheSame<IntervalsAlignmentAttribute>(operations))
                << "IntervalsAlignmentAttribute are not the same";
        }
    }
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

namespace testValues1 {
const std::vector<std::vector<ngraph::PartialShape>> shapes = {{{1, 16}, {1, 128}, {1, 128}}};

const std::vector<LSTMTransformationTestValues> testValues = {
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {{{256ul, {}, {0.f}, {2550.f}, {0.f}, {2550.f}}}, {{}}, {{}}},
     {{{256ul,
        {},
        {0.f},
        {2550.f},
        {0.f},
        {255.f},
        ngraph::element::u8,
        {IntervalsAlignmentAttribute(IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}}},
      {{}},
      {{}}},
     true},
};
INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    LSTMTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    LSTMTransformation::getTestCaseName);
}  // namespace testValues1
}  // namespace