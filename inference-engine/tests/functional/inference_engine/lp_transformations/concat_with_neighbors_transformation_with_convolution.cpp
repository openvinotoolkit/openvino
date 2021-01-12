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

#include <low_precision/rt_info/intervals_alignment_attribute.hpp>
#include <low_precision/rt_info/quantization_alignment_attribute.hpp>
#include <low_precision/rt_info/precisions_attribute.hpp>
#include <low_precision/rt_info/precision_preserved_attribute.hpp>

#include <low_precision/align_concat_quantization_parameters.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_avg_pool_precisions.hpp>
#include <low_precision/propagate_precisions.hpp>

#include <low_precision/concat.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/max_pool.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
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
    ngraph::pass::low_precision::LayerTransformation::Params params;
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

class ConcatWithNeighborsWithConvolutionTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatWithNeighborsWithConvolutionParams> {
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

#define VISUALIZE_TREE
#ifndef VISUALIZE_TREE

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisionsOnActivation);
        manager.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisions>();
        manager.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
        manager.register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();

        std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
        common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>();
        common->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>();
        common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
        common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>();

        manager.run_passes(actualFunction);

#else
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.actual").run_on_function(actualFunction);

        {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisionsOnActivation);
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming1").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisions>();
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming2").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming3").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming4").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager;
            std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
            common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transformed").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager;
            std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
            common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>();
            common->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>();
            common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>();

            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transformed").run_on_function(actualFunction);
        }
#endif

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

#ifdef VISUALIZE_TREE
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.reference").run_on_function(referenceFunction);
#endif
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
    auto res = compare_functions(referenceFunction, actualFunction, true, false, false);
    ASSERT_TRUE(res.first) << res.second;

    const auto actualFakeQuantizes = LayerTransformation::get<opset1::FakeQuantize>(actualFunction);
    ASSERT_TRUE(checkIfOutputAttributesAreTheSame<std::shared_ptr<PrecisionsAttribute>>(actualFakeQuantizes)) << "PrecisionsAttribute are not the same";

    const auto referenceFakeQuantizes = LayerTransformation::get<opset1::FakeQuantize>(referenceFunction);
    // TODO: not completed
    //ASSERT_TRUE(checkIfOutputAttributesAreEqual<std::shared_ptr<IntervalsAlignmentAttribute>>(actualFakeQuantizes, referenceFakeQuantizes)) <<
    //    "attributes are not the equal";

    auto operations = LayerTransformation::get<opset1::Concat>(actualFunction);
    operations.insert(operations.end(), actualFakeQuantizes.begin(), actualFakeQuantizes.end());
    ASSERT_TRUE(checkIfAttributesAreTheSame<std::shared_ptr<IntervalsAlignmentAttribute>>(operations)) << "IntervalsAlignmentAttribute are not the same";
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
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
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 1.27f) }
            },
            {
                256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {64.f}, {192.f}, element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 1.27f) }
            },
            {
                256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f}, element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 1.27f) }
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

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    ConcatWithNeighborsWithConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatWithNeighborsWithConvolutionTransformation::getTestCaseName);
}  // namespace
