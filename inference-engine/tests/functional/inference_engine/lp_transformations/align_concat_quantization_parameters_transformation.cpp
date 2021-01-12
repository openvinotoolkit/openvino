// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include <low_precision/align_quantization_intervals.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_avg_pool_precision_preserved.hpp>
#include <low_precision/propagate_precisions.hpp>

#include <low_precision/avg_pool.hpp>
#include <low_precision/concat.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/max_pool.hpp>
#include <low_precision/align_quantization_parameters.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/align_concat_quantization_parameters_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph::pass;

class AlignConcatQuantizationParametersTransformationTestValues {
public:
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool, // additional FakeQuantize After
    std::string, // additional layer before FQ
    AlignConcatQuantizationParametersTransformationTestValues> AlignConcatQuantizationParametersTransformationParams;

class AlignConcatQuantizationParametersTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<AlignConcatQuantizationParametersTransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AlignConcatQuantizationParametersTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = GetParam();

        actualFunction = ngraph::builder::subgraph::AlignConcatQuantizationParametersFunction::getOriginal(
            precision,
            testValues.actual.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.actual.dequantization);

        auto supportedPrecisions = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
            ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            })
        });

        auto perTensorQuantization = std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>({
            ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction::create<ngraph::opset1::Convolution>({0}),
        });

//#define VISUALIZE_TREE
#ifndef VISUALIZE_TREE
        SimpleLowPrecisionTransformer transform(supportedPrecisions, perTensorQuantization);
        transform.add<ngraph::pass::low_precision::AvgPoolTransformation, ngraph::opset1::AvgPool>(testValues.params);
        transform.add<ngraph::pass::low_precision::ConcatTransformation, ngraph::opset1::Concat>(testValues.params);
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);
#else
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.actual.svg").run_on_function(actualFunction);

        {
            ngraph::pass::Manager manager1;
            manager1.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisions);
            manager1.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming1.svg").run_on_function(actualFunction);
            //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming1").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager2;
            manager2.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
            manager2.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming2.svg").run_on_function(actualFunction);
            //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming2").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager3;
            manager3.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
            manager3.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming3.svg").run_on_function(actualFunction);
            //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming3").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager4;
            manager4.register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
            manager4.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming4.svg").run_on_function(actualFunction);
            //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming4").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager5;
            manager5.register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();
            manager5.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming5.svg").run_on_function(actualFunction);
            //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming5").run_on_function(actualFunction);
        }

        {
            ngraph::pass::Manager manager;
            std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
            common->add_matcher<low_precision::AvgPoolTransformation>();
            common->add_matcher<low_precision::ConcatTransformation>();
            common->add_matcher<low_precision::ConvolutionTransformation>();
            common->add_matcher<low_precision::FakeQuantizeDecompositionTransformation>();
            common->add_matcher<low_precision::MaxPoolTransformation>();
            manager.run_passes(actualFunction);
            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transformed.svg").run_on_function(actualFunction);
            //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transformed").run_on_function(actualFunction);
        }
#endif
        referenceFunction = ngraph::builder::subgraph::AlignConcatQuantizationParametersFunction::getReference(
            precision,
            testValues.expected.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.expected.dequantizationBefore,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);

#ifdef VISUALIZE_TREE
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.reference.svg").run_on_function(referenceFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.reference").run_on_function(actualFunction);
#endif
    }

    static std::string getTestCaseName(testing::TestParamInfo<AlignConcatQuantizationParametersTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AlignConcatQuantizationParametersTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = obj.param;

        std::ostringstream result;
        result <<
            precision << "_" <<
            LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision, shape, testValues.params) << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.preicsionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter << "_" <<
            (addFakeQuantize ? "_FQ_after_" : "_") << additionalLayer;
        return result.str();
    }
};

TEST_P(AlignConcatQuantizationParametersTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<std::string> additionalLayer = {
    "maxpool"  // any transparent layer
};

const std::vector<bool> addFQ = {
    // true,
    false
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 }
};

const std::vector<AlignConcatQuantizationParametersTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::f32,
            {{}, {std::vector<float>(6, 128.f), element::f32, {1, 6, 1, 1}}, {}},
            ngraph::element::f32,
            {{}, {}, {std::vector<float>(9, 0.0001f), element::f32, {1, 9, 1, 1}}}
        }
    }
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    AlignConcatQuantizationParametersTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(addFQ),
        ::testing::ValuesIn(additionalLayer),
        ::testing::ValuesIn(testValues)),
    AlignConcatQuantizationParametersTransformation::getTestCaseName);
