// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <low_precision/concat.hpp>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/relu.hpp>

#include <low_precision/low_precision.hpp>

#include "low_precision/move_fake_quantize.hpp"
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/rt_info/precision_preserved_attribute.hpp>
#include <low_precision/align_quantization_parameters.hpp>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include <low_precision/fuse_multiply_to_fake_quantize.hpp>
#include <low_precision/markup_can_be_quantized.hpp>
#include <low_precision/markup_per_tensor_quantization.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/move_fake_quantize_function.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/relu_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

    class MoveFakeQuantizeActualValues {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeBefore1;
        ngraph::builder::subgraph::DequantizationOperations::Convert convertBefore1;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore1;
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeBefore2;
        ngraph::builder::subgraph::DequantizationOperations::Convert convertBefore2;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore2;
        std::string operation;
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeAfter;
        ngraph::builder::subgraph::DequantizationOperations::Convert convertAfter;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeActualValues& values) {
        return out << "_" <<
            values.fakeQuantizeBefore1 << "_" <<
            values.convertBefore1.outPrecision << "_" <<
            values.dequantizationBefore1 << "_" <<
            values.fakeQuantizeBefore2 << "_" <<
            values.convertBefore2.outPrecision << "_" <<
            values.dequantizationBefore2 << "_" <<
            values.operation << "_" <<
            values.fakeQuantizeAfter << "_" <<
            values.convertAfter.outPrecision << "_" <<
            values.dequantizationAfter;
    }

    class MoveFakeQuantizeResultValues {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeBefore1;
        ngraph::builder::subgraph::DequantizationOperations::Convert convertBefore1;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore1;
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeBefore2;
        ngraph::builder::subgraph::DequantizationOperations::Convert convertBefore2;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore2;
        std::string operation;
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeAfter;
        ngraph::builder::subgraph::DequantizationOperations::Convert convertAfter;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfterNotFQ;
    };

    inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeResultValues& values) {
        return out << "_" <<
            values.fakeQuantizeBefore1 << "_" <<
            values.convertBefore1.outPrecision << "_" <<
            values.dequantizationBefore1 << "_" <<
            values.fakeQuantizeBefore2 << "_" <<
            values.convertBefore2.outPrecision << "_" <<
            values.dequantizationBefore2 << "_" <<
            values.operation << "_" <<
            values.fakeQuantizeAfter << "_" <<
            values.convertAfter << "_" <<
            values.dequantizationAfter << "_" <<
            values.dequantizationAfterNotFQ;
    }

    class MoveFakeQuantizeTestValues {
    public:
        MoveFakeQuantizeTestValues() = default;
        MoveFakeQuantizeTestValues(
            const TestTransformationParams& params,
            const bool multiChannels,
            const  std::int64_t axis,
            const MoveFakeQuantizeActualValues& actual,
            const MoveFakeQuantizeResultValues& result,
            const bool addNotPrecisionPreservedOperation = false,
            const bool checkIntervalsAlignmentAttributes = true) :
            params(params),
            multiChannels(multiChannels),
            axis(axis),
            actual(actual),
            result(result),
            addNotPrecisionPreservedOperation(addNotPrecisionPreservedOperation),
            checkIntervalsAlignmentAttributes(checkIntervalsAlignmentAttributes) {}

        TestTransformationParams params;
        bool multiChannels;
        std::int64_t axis;
        MoveFakeQuantizeActualValues actual;
        MoveFakeQuantizeResultValues result;
        // add not precision preserved operation to set output precision for FakeQuantize
        // don't set to 'true' by default to keep test cases with tested operation as output
        bool addNotPrecisionPreservedOperation;
        bool checkIntervalsAlignmentAttributes;
    };

    inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeTestValues& values) {
        return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
    }

    typedef std::tuple <
        ngraph::element::Type,
        ngraph::PartialShape,
        MoveFakeQuantizeTestValues
    > MoveFakeQuantizeParams;

    class MoveFakeQuantize : public LayerTransformation, public testing::WithParamInterface<MoveFakeQuantizeParams> {
    public:
        void SetUp() override {
            const ngraph::element::Type precision = std::get<0>(GetParam());
            const ngraph::PartialShape shape = std::get<1>(GetParam());
            MoveFakeQuantizeTestValues testValues = std::get<2>(GetParam());

            // dequantization output precision depends on input precision
            // to avoid huge amount of tests cases let's define dequantization output precision as input precision
            if (!testValues.actual.dequantizationBefore1.multiply.empty()) {
                testValues.actual.dequantizationBefore1.multiply.outPrecision = precision;
            }
            if (!testValues.actual.dequantizationBefore2.multiply.empty()) {
                testValues.actual.dequantizationBefore2.multiply.outPrecision = precision;
            }
            actualFunction = ngraph::builder::subgraph::MoveFakeQuantize::get(
                precision,
                shape,
                testValues.actual.fakeQuantizeBefore1,
                testValues.actual.convertBefore1,
                testValues.actual.dequantizationBefore1,
                testValues.actual.fakeQuantizeBefore2,
                testValues.actual.convertBefore2,
                testValues.actual.dequantizationBefore2,
                testValues.actual.operation,
                testValues.actual.fakeQuantizeAfter,
                testValues.actual.convertAfter,
                testValues.actual.dequantizationAfter,
                ngraph::element::undefined,
                {},
                testValues.axis);
            ngraph::pass::VisualizeTree("c:\\Users\\ndemasho\\rep\\Visual\\MFQtest.actual").run_on_function(actualFunction);

            auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
                    ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::AvgPool>({{0, testValues.params.precisionsOnActivations}})
                });

            auto quantizationRestrictions = testValues.multiChannels ?
                std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>() :
                std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>({
                    ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction::create<ngraph::opset1::AvgPool>()
                    });

            const auto params = TestTransformationParams::toParams(testValues.params);
            SimpleLowPrecisionTransformer transformer(supportedPrecisionsOnActivation, quantizationRestrictions);
            //SimpleLowPrecisionTransformer transformer({}, {});
            transformer.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::MoveFakeQuantize>(params);
            transformer.transform(actualFunction);
            ngraph::pass::VisualizeTree("c:\\Users\\ndemasho\\rep\\Visual\\MFQtest.transform").run_on_function(actualFunction);

            // dequantization output precision depends on input precision
            // to avoid huge amount of tests cases let's define dequantization output precision as input precision
            if (!testValues.result.dequantizationAfter.multiply.empty()) {
                testValues.result.dequantizationAfter.multiply.outPrecision = precision;
            }

            if (!testValues.params.updatePrecisions &&
                (precision == ngraph::element::f32) &&
                !testValues.result.dequantizationAfter.convert.empty()) {
                testValues.result.dequantizationAfter.convert = {};
            }

            //IntervalsAlignmentSharedValue::Interval interval{ -1.28f, 2.55f };

            referenceFunction = ngraph::builder::subgraph::MoveFakeQuantize::get(
                precision,
                shape,
                testValues.result.fakeQuantizeBefore1,
                testValues.result.convertBefore1,
                testValues.result.dequantizationBefore1,
                testValues.result.fakeQuantizeBefore2,
                testValues.result.convertBefore2,
                testValues.result.dequantizationBefore2,
                testValues.result.operation,
                testValues.result.fakeQuantizeAfter,
                testValues.result.convertAfter,
                testValues.result.dequantizationAfter,
                testValues.result.precisionAfterOperation,
                {},
                testValues.axis);
            ngraph::pass::VisualizeTree("c:\\Users\\ndemasho\\rep\\Visual\\MFQtest.reference").run_on_function(referenceFunction);
        }

        static std::string getTestCaseName(testing::TestParamInfo<MoveFakeQuantizeParams> obj) {
            const ngraph::element::Type precision = std::get<0>(obj.param);
            const ngraph::PartialShape shape = std::get<1>(obj.param);
            const MoveFakeQuantizeTestValues testValues = std::get<2>(obj.param);

            std::ostringstream result;
            result <<
                LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
                (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
                "axis_" << testValues.axis << "_" <<
                testValues.actual << "_" <<
                testValues.result << "_";
            return result.str();
        }
    };

    TEST_P(MoveFakeQuantize, CompareFunctions) {
        actualFunction->validate_nodes_and_infer_types();
        auto res = compare_functions(referenceFunction, actualFunction, true, true, false, true, false);
        ASSERT_TRUE(res.first) << res.second;

        const auto actualFakeQuantizes = LayerTransformation::get<opset1::FakeQuantize>(actualFunction);
        ASSERT_TRUE(checkIfOutputAttributesSharedValuesAreTheSame<std::shared_ptr<PrecisionsAttribute>>(actualFakeQuantizes)) <<
            "PrecisionsAttribute are not the same";

        MoveFakeQuantizeTestValues testValues = std::get<2>(GetParam());
        if (testValues.checkIntervalsAlignmentAttributes) {
            auto operations = LayerTransformation::get<opset1::FakeQuantize>(actualFunction);
            operations.insert(operations.end(), actualFakeQuantizes.begin(), actualFakeQuantizes.end());
            ASSERT_TRUE(checkIfAttributesSharedValuesAreTheSame<std::shared_ptr<IntervalsAlignmentAttribute>>(operations)) <<
                "IntervalsAlignmentAttribute are not the same";
        }
    }

    const std::vector<ngraph::element::Type> precisions = {
        ngraph::element::f32,
        //ngraph::element::f16
    };

    namespace testValues1 {
        const std::vector<ngraph::PartialShape> shapes = {
            { 1, 3, 9, 9 },
            //{ 4, 3, 9, 9 },
            //{ Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() }
        };

        const std::vector<MoveFakeQuantizeTestValues> testValues = {
            // U8: concat
            {
                        LayerTransformation::createParamsU8I8(),
                        false,
                        1,
                        {
                            {},
                            {},
                            {},
                            {},
                            {},
                            {},
                            "",
                            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
                            {},
                            {}
                        },
                        {
                            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
                            {},
                            {},
                            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
                            {},
                            {},
                            "",
                            {},
                            {},
                            {},
                        },
                        false,
                        false
            },
            {
                        LayerTransformation::createParamsU8I8(),
                        false,
                        1,
                        {
                            {},
                            {},
                            {},
                            {},
                            {},
                            {},
                            "relu",
                            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
                            {},
                            {}
                        },
                        {
                            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
                            {},
                            {},
                            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
                            {},
                            {},
                            "relu",
                            {},
                            {},
                            {},
                        },
                        false,
                        false
             }
        };
        INSTANTIATE_TEST_SUITE_P(
            smoke_LPT,
            MoveFakeQuantize,
            ::testing::Combine(
                ::testing::ValuesIn(precisions),
                ::testing::ValuesIn(shapes),
                ::testing::ValuesIn(testValues)),
            MoveFakeQuantize::getTestCaseName);
    } // namespace testValues1
} // namespace
