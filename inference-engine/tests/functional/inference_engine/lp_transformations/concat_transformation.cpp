// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <sstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include <low_precision/low_precision.hpp>

#include <low_precision/concat.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/rt_info/precision_preserved_attribute.hpp>
#include <low_precision/align_quantization_parameters.hpp>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include <low_precision/fuse_multiply_to_fake_quantize.hpp>
#include <low_precision/markup_per_tensor_quantization.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

namespace {

class ConcatTransformationActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2;
}

class ConcatTransformationResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2 << "_" <<
        values.dequantizationAfter;
}

class ConcatTransformationTestValues {
public:
    ConcatTransformationTestValues() = default;
    ConcatTransformationTestValues(
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const bool multiChannels,
        const  std::int64_t axis,
        const ConcatTransformationActualValues& actual,
        const ConcatTransformationResultValues& result,
        const bool addNotPrecisionPreservedOperation = false,
        const bool checkIntervalsAlignmentAttributes = true) :
        params(params),
        multiChannels(multiChannels),
        axis(axis),
        actual(actual),
        result(result),
        addNotPrecisionPreservedOperation(addNotPrecisionPreservedOperation),
        checkIntervalsAlignmentAttributes(checkIntervalsAlignmentAttributes) {}

    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannels;
    std::int64_t axis;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
    // add not precision preserved operation to set output precision for FakeQuantize
    // don't set to 'true' by default to keep test cases with tested operation as output
    bool addNotPrecisionPreservedOperation;
    bool checkIntervalsAlignmentAttributes;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::Shape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.actual.dequantization1.multiply.empty()) {
            testValues.actual.dequantization1.multiply.outPrecision = precision;
        }
        if (!testValues.actual.dequantization2.multiply.empty()) {
            testValues.actual.dequantization2.multiply.outPrecision = precision;
        }

        actualFunction = ngraph::builder::subgraph::ConcatFunction::get(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.convert1,
            testValues.actual.dequantization1,
            testValues.actual.fakeQuantize2,
            testValues.actual.convert2,
            testValues.actual.dequantization2,
            {},
            ngraph::element::undefined,
            {},
            testValues.axis,
            testValues.addNotPrecisionPreservedOperation);

        auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
            ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            }),
            ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::AvgPool>({{0, testValues.params.precisionsOnActivations}})
        });

        auto quantizationRestrictions = testValues.multiChannels ?
            std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>() :
            std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>({
                ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction::create<ngraph::opset1::AvgPool>()
            });

        const auto params = ngraph::pass::low_precision::LayerTransformation::Params(testValues.params.updatePrecisions);

//#define VISUALIZE_TREE
#ifndef VISUALIZE_TREE
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisionsOnActivation);
        manager.register_pass<ngraph::pass::low_precision::MarkupPerTensorQuantization>(quantizationRestrictions);
        manager.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
        manager.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
        manager.register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
        manager.register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();

        std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
        common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>(params);
        common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>(params);
        manager.run_passes(actualFunction);

        {
            ngraph::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }
#else
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.actual.svg").run_on_function(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.actual").run_on_function(actualFunction);

        ngraph::pass::Manager manager1;
        manager1.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisionsOnActivation);
        manager1.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming1.svg").run_on_function(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming1").run_on_function(actualFunction);

        ngraph::pass::Manager manager12;
        manager12.register_pass<ngraph::pass::low_precision::MarkupPerTensorQuantization>(quantizationRestrictions);
        manager12.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming1_2.svg").run_on_function(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming1_2").run_on_function(actualFunction);

        ngraph::pass::Manager manager2;
        manager2.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
        manager2.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming2.svg").run_on_function(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming2").run_on_function(actualFunction);

        ngraph::pass::Manager manager3;
        manager3.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
        manager3.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming3.svg").run_on_function(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming3").run_on_function(actualFunction);

        ngraph::pass::Manager manager4;
        manager4.register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
        manager4.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming4.svg").run_on_function(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming4").run_on_function(actualFunction);

        ngraph::pass::Manager manager5;
        manager4.register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();
        manager4.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transforming5.svg").run_on_function(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming5").run_on_function(actualFunction);

        {
            ngraph::pass::Manager manager;
            std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
            common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>(params);
            common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>(params);
            manager.run_passes(actualFunction);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        {
            ngraph::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager.register_pass<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transformed.svg").run_on_function(actualFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transformed").run_on_function(actualFunction);
#endif
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

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::get(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.convert1,
            testValues.result.dequantization1,
            testValues.result.fakeQuantize2,
            testValues.result.convert2,
            testValues.result.dequantization2,
            {
                make_shared_attribute_ptr<PrecisionPreservedAttribute>(true),
                make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 2.55f),
                make_shared_attribute_ptr<QuantizationAlignmentAttribute>(false)
            },
            testValues.result.precisionAfterOperation,
            testValues.result.dequantizationAfter,
            testValues.axis,
            testValues.addNotPrecisionPreservedOperation);

#ifdef VISUALIZE_TREE
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.reference.svg").run_on_function(referenceFunction);
        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.reference").run_on_function(referenceFunction);
#endif
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);

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

TEST_P(ConcatTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, false, true, false);
    ASSERT_TRUE(res.first) << res.second;

    const auto actualFakeQuantizes = LayerTransformation::get<opset1::FakeQuantize>(actualFunction);
    ASSERT_TRUE(checkIfOutputAttributesSharedValuesAreTheSame<std::shared_ptr<PrecisionsAttribute>>(actualFakeQuantizes)) <<
        "PrecisionsAttribute are not the same";

    ConcatTransformationTestValues testValues = std::get<2>(GetParam());
    if (testValues.checkIntervalsAlignmentAttributes) {
        auto operations = LayerTransformation::get<opset1::Concat>(actualFunction);
        operations.insert(operations.end(), actualFakeQuantizes.begin(), actualFakeQuantizes.end());
        ASSERT_TRUE(checkIfAttributesSharedValuesAreTheSame<std::shared_ptr<IntervalsAlignmentAttribute>>(operations)) <<
            "IntervalsAlignmentAttribute are not the same";
    }
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} }
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } },
        }
    },
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {}
        },
        {
            {
                256ul, {{1}, {1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {{1}, {1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {}
        },
        {
            {
                256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // U8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {0.f}, {1.275f}, {0.f}, {1.275f} },
            {},
            {}
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {0.f}, {1.275f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} }
        }
    },
    // U8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {1.275f}, {0.f}, {1.275f} },
            {},
            {}
        },
        {
            {
                256ul, {{1}, {1}, {}, {}}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {{1}, {1}, {}, {}}, {0.f}, {1.275f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} }
        }
    },
    // U8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                {0.f, 0.f, 0.f}, {2.55f, 2.55f, 2.55f}, {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}
            },
            {},
            {},
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                {0.f, 0.f, 0.f}, {1.275f, 1.275f, 1.275f}, {0.f, 0.f, 0.f}, {1.275f / 1.f, 1.275f / 2.f, 1.275f / 3.f}
            },
            {},
            {}
        },
        {
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                {0.f, 0.f, 0.f}, {2.55f, 2.55f, 2.55f}, {0.f}, {255.f},
                ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                {0.f, 0.f, 0.f}, {1.275f, 1.275f, 1.275f}, {0.f}, {255.f},
                ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.01f / 1.f, 0.01f / 2.f, 0.01f / 3.f, 0.005f / 1.f, 0.005f / 2.f, 0.005f / 3.f }} }
        }
    },
    // U8: concat multi channels with subtract
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            { 256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {}
        },
        {
            {
                256ul, {}, {1.275f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            {
                ngraph::element::f32,
                {{ -255.f, -255.f, -255.f, 0.f, 0.f, 0.f }},
                {{ 0.005f, 0.005f, 0.005f, 0.01f, 0.01f, 0.01f }}
            }
        }
    },
    // I8
    {
        LayerTransformation::createParamsI8I8(),
        false,
        1,
        {
            { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            {},
            {},
            { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            {},
            {}
        },
        {
            {
                256ul, {}, {-1.28f}, {1.27f}, {-128.f}, {127.f}, ngraph::element::i8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {-1.28f}, {1.27f}, {-128.f}, {127.f}, ngraph::element::i8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(0.f, 2.55f) }
            },
            {},
            {},
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // mixed: U8 + I8: concat (check constant values here)
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            {},
            {}
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, { {0.f, 0.f, 0.f, 128.f, 128.f, 128.f } }, { 0.01f } }
        }
    },
    // mixed: U8 + I8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            {},
            {}
        },
        {
            {
                256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 2.55f) }
            },
            {},
            {},
            {
                256ul, {}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ngraph::element::u8,
                { make_shared_attribute_ptr<IntervalsAlignmentAttribute>(-1.28f, 2.55f) }
            },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {{ 0.f, 0.f, 0.f, 128.f, 128.f, 128.f }}, { 0.01f } }
        }
    },
    // mixed: I8 + U8: concat (check constant values here)
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {}
        },
        {
            { 256ul, {}, {-1.28f}, {1.27f}, {0.f}, {170.f}, ngraph::element::u8 },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {85.f}, {255.f}, ngraph::element::u8 },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, { 85 }, { 0.015f } }
        },
        true
    },
    // real case from ctdet_coco_dlav0_384 model, coverage bad rounding
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {}, {-1.28f}, {1.27f}, {0.f}, {2.3007815f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {-3.873046875f}, {3.84375} },
            {},
            {}
        },
        {
            { 256ul, {}, {-1.28f}, {1.27f}, {128.f}, {204.f}, ngraph::element::u8 },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            {},
            {},
            ngraph::element::u8,
            { ngraph::element::f32, { 128 }, { 0.0302619f } }
        },
        true
    },
    // U8: concat multi channels with subtract, negative axis
    {
        LayerTransformation::createParamsU8I8(),
        true,
        -3,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f} },
            {},
            {}
        },
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            {},
            {},
            { 256ul, {}, {1.275f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            {},
            {},
            ngraph::element::u8,
            {
                ngraph::element::f32,
                {{ 0.f, 0.f, 0.f, -255.f, -255.f, -255.f }},
                {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }}
            }
        }
    },
    // U8: concat multi channels with subtract, not supported axis
    {
        LayerTransformation::createParamsU8I8(),
        true,
        0,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f} },
            {},
            {}
        },
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f} },
            {},
            {}
        },
    },
    // not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        false,
        1,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {}
        },
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            {},
            {},
            ngraph::element::f32,
            { {element::f32}, {}, { 0.01f } },
        }
    },
    // unexpected quantization levels, concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 16ul, {}, {0.f}, {1.5f}, {0.f}, {15.f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {}
        },
        {
            { 16ul, {}, {0.f}, {1.5f}, {0.f}, {15.f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            ngraph::element::f32,
            {},
        },
        false,
        false,
    },
    // unexpected quantization levels, concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            { 16ul, {}, {0.f}, {1.5f}, {0.f}, {15.f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {}
        },
        {
            { 16ul, {}, {0.f}, {1.5f}, {0.f}, {15.f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            ngraph::element::f32,
            {},
        },
        false,
        false
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 },
    { 4, 3, 9, 9 }
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatTransformation::getTestCaseName);
}  // namespace
