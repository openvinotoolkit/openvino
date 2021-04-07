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
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class ConcatWithNeighborsWithConvolutionActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
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
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3);

        auto supportedPrecisionsOnActivation = std::vector<OperationPrecisionRestriction>({
            OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            })
        });

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
    auto res = compare_functions(referenceFunction, actualFunction, true, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ConcatWithNeighborsWithConvolutionTestValues> testValues = {
    // I8: concat
    {
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
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
