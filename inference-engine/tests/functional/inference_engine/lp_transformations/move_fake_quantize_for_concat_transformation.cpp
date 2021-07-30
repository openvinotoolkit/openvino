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
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize3;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert3;
    ngraph::builder::subgraph::DequantizationOperations dequantization3;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeActualValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2;
}

class MoveFakeQuantizeResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize3;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert3;
    ngraph::builder::subgraph::DequantizationOperations dequantization3;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2 << "_" <<
        values.dequantizationAfter;
}

class MoveFakeQuantizeTestValues {
public:
    MoveFakeQuantizeTestValues() = default;
    MoveFakeQuantizeTestValues(
        const TestTransformationParams & params,
        const bool multiChannels,
        const  std::int64_t axis,
        const MoveFakeQuantizeActualValues & actual,
        const MoveFakeQuantizeResultValues & result,
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
        if (!testValues.actual.dequantization1.multiply.empty()) {
            testValues.actual.dequantization1.multiply.outPrecision = precision;
        }
        if (!testValues.actual.dequantization2.multiply.empty()) {
            testValues.actual.dequantization2.multiply.outPrecision = precision;
        }
        actualFunction = ngraph::builder::subgraph::MoveFakeQuantize::get(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.convert1,
            testValues.actual.dequantization1,
            testValues.actual.fakeQuantize2,
            testValues.actual.convert2,
            testValues.actual.dequantization2,
            testValues.actual.fakeQuantize3,
            testValues.actual.convert3,
            testValues.actual.dequantization3,
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
            testValues.result.fakeQuantize1,
            testValues.result.convert1,
            testValues.result.dequantization1,
            testValues.result.fakeQuantize2,
            testValues.result.convert2,
            testValues.result.dequantization2,
            testValues.result.fakeQuantize3,
            testValues.result.convert3,
            testValues.result.dequantization3,
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
                    {},
                    {},
                    {},
                },
                false,
                false
            },
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
