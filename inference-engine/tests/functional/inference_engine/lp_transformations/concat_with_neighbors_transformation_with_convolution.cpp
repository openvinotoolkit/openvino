// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <sstream>
#include <memory>
#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <low_precision/concat.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/max_pool.hpp>

#include "lpt_ngraph_functions/precision_propagation_function.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

namespace {

class ConcatWithNeighborsWithConvolutionActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert3;
    ngraph::builder::subgraph::DequantizationOperations dequantization3;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatWithNeighborsWithConvolutionActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.fakeQuantize3;
}

class ConcatWithNeighborsWithConvolutionResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
    ngraph::element::Type precisionBeforeOp;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
    ngraph::element::Type precisionAfterOp;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter1;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatWithNeighborsWithConvolutionResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.fakeQuantize3 << "_" <<
        values.dequantizationAfter1 << "_" <<
        values.dequantizationAfter2;
}

class ConcatWithNeighborsWithConvolutionTestValues {
public:
    TestTransformationParams params;
    bool multiChannels;
    ConcatWithNeighborsWithConvolutionActualValues actual;
    ConcatWithNeighborsWithConvolutionResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatWithNeighborsWithConvolutionTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::Shape,
    ConcatWithNeighborsWithConvolutionTestValues
> ConcatWithNeighborsWithConvolutionParams;

class ConcatWithNeighborsWithConvolutionTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<ConcatWithNeighborsWithConvolutionParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        ConcatWithNeighborsWithConvolutionTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::PrecisionPropagationFunction::getOriginalWithNeighbors(
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
            testValues.actual.dequantization3);

        auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
            ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            })
        });

        auto quantizationRestrictions = testValues.multiChannels ?
            std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>() :
            std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>({
                ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction::create<ngraph::opset1::Convolution>({0})
            });

        SimpleLowPrecisionTransformer transform(supportedPrecisionsOnActivation, quantizationRestrictions);
        transform.add<ngraph::pass::low_precision::ConcatTransformation, ngraph::opset1::Concat>(testValues.params);
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::PrecisionPropagationFunction::getReferenceWithNeighbors(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.fakeQuantize3,
            testValues.result.precisionBeforeOp,
            testValues.result.dequantizationBefore,
            testValues.result.precisionAfterOp,
            testValues.result.dequantizationAfter1,
            testValues.result.dequantizationAfter2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatWithNeighborsWithConvolutionParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const ConcatWithNeighborsWithConvolutionTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithNeighborsWithConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    //auto res = compare_functions(referenceFunction, actualFunction, true, false, false);
    //ASSERT_TRUE(res.first) << res.second;

    auto actualFakeQuantizes = LayerTransformation::get<opset1::FakeQuantize>(actualFunction);
    ASSERT_EQ(3ul, actualFakeQuantizes.size()) << "unexpected FakeQuantize operations count " << actualFakeQuantizes.size();

    ASSERT_TRUE(checkIfOutputAttributesSharedValuesAreTheSame<std::shared_ptr<PrecisionsAttribute>>(actualFakeQuantizes)) <<
        "PrecisionsAttribute shared values are not the same";

    auto actualConcatOperations = LayerTransformation::get<opset1::Concat>(actualFunction);
    ASSERT_EQ(2ul, actualConcatOperations.size()) << "unexpected concat operations";
    ASSERT_NE(nullptr, ngraph::pass::low_precision::getAttribute<std::shared_ptr<QuantizationAlignmentAttribute>>(actualConcatOperations[0]));
    ASSERT_NE(nullptr, ngraph::pass::low_precision::getAttribute<std::shared_ptr<QuantizationAlignmentAttribute>>(actualConcatOperations[1]));

    actualConcatOperations.insert(actualConcatOperations.end(), actualFakeQuantizes.begin(), actualFakeQuantizes.end());
    ASSERT_TRUE(checkIfAttributesSharedValuesAreTheSame<std::shared_ptr<IntervalsAlignmentAttribute>>(actualConcatOperations)) <<
        "IntervalsAlignmentAttribute shared values are not the same";

    auto convolutions = LayerTransformation::get<opset1::Convolution>(actualFunction);
    ASSERT_EQ(1ul, convolutions.size()) << "unexpected convolution operations";
    ASSERT_EQ(2ul, convolutions[0]->input(0).get_rt_info().size()) <<
        "unexpected input 0 attributes count: LowPrecision::PerTensorQuantization & LowPrecision::Precisions";
    ASSERT_EQ(1ul, convolutions[0]->input(1).get_rt_info().size()) << "unexpected input 1 attributes count";
    auto a1 = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(convolutions[0]->input(1).get_rt_info().begin()->second);
    ASSERT_EQ(element::i8, *a1->get().get()->sharedValue->precisions.begin());
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32
};

const std::vector<ConcatWithNeighborsWithConvolutionTestValues> testValues = {
    // I8: concat: composed FakeQuantize
    {
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} },
            {},
            {},
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            {},
            {},
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            {},
            {}
        },
        {
            {
                256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {0.f}, {255.f}, element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(IntervalsAlignmentSharedValue::Interval{-1.28f, 1.27f}, 256ul) }
            },
            {
                256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {64.f}, {192.f}, element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(IntervalsAlignmentSharedValue::Interval{-1.28f, 1.27f}, 256ul) }
            },
            {
                256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f}, element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(IntervalsAlignmentSharedValue::Interval{-1.28f, 1.27f}, 256ul) }
            },
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {128.f}, {{ 0.00333333f, 0.00333333f, 0.00333333f, 0.01f, 0.01f, 0.01f }} },
            { {}, {}, {{ 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f }} }
        }
    },
    // I8: concat: decomposed FakeQuantize
    {
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-128.f}, {127.f} },
            { ngraph::element::i8 },
            {
                { element::f32 },
                {},
                { 0.003333333333333f }
            },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            {},
            {},
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            {},
            {}
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {64.f}, {192.f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {128.f}, {{ 0.00333333f, 0.00333333f, 0.00333333f, 0.01f, 0.01f, 0.01f }} },
            { {}, {}, {{ 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.0001f }} }
        }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 },
    { 4, 3, 9, 9 }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithNeighborsWithConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatWithNeighborsWithConvolutionTransformation::getTestCaseName);
}  // namespace
