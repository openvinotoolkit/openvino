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
#include <low_precision/concat.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/split.hpp>
#include <low_precision/align_quantization_parameters.hpp>
#include <low_precision/align_quantization_intervals.hpp>
#include <low_precision/propagate_precisions.hpp>
#include <low_precision/markup_avg_pool_precision_preserved.hpp>
#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_per_tensor_quantization.hpp>
#include "low_precision/common/operation_precision_restriction.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"


using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class ConcatTransformationActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2;
}

class ConcatTransformationResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::element::Type precisionBeforeOp;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore1;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore2;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations1;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.dequantizationOperations1 << "_" <<
        values.dequantizationOperations2;
}

class ConcatTransformationTestValues {
public:
    TestTransformationParams params;
    bool multiChannels;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::PartialShape,
    ConcatTransformationTestValues,
    bool // additional Convolution after Split
> ConcatTransformationParams;

class ConcatWithSplitTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::PartialShape shape = std::get<1>(GetParam());
        const ConcatTransformationTestValues testValues = std::get<2>(GetParam());
        const bool addConvolution = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginalWithSplitedIntermediate(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            addConvolution);

        auto supportedPrecisions = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
               ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                   {0, testValues.params.precisionsOnActivations},
                   {1, testValues.params.precisionsOnWeights},
               })
           });

        auto quantizationRestrictions = testValues.multiChannels ?
            std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>() :
            std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>({
                ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction::create<ngraph::opset1::Convolution>()
            });

        SimpleLowPrecisionTransformer transform(supportedPrecisions, quantizationRestrictions);
        transform.add<ngraph::pass::low_precision::ConcatTransformation, ngraph::opset1::Concat>(testValues.params);
        transform.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.add<ngraph::pass::low_precision::SplitTransformation, ngraph::opset1::Split>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReferenceWithSplitedIntermediate(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.precisionBeforeOp,
            testValues.result.dequantizationBefore1,
            testValues.result.dequantizationBefore2,
            testValues.result.precisionAfterOperation,
            addConvolution,
            testValues.result.dequantizationOperations1,
            testValues.result.dequantizationOperations2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::PartialShape shape = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);
        const bool addConvolution = std::get<3>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            (addConvolution ? "" : "without_convolution_") <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithSplitTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::PartialShape> shapes = {
    { 1, 6, 10, 10 },
    { Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic() }
};

namespace casesWithConvolution {
const std::vector<ConcatTransformationTestValues> testValues = {
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}},
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, { 255.f}},
            ngraph::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, { 0.005f } }
        }
    },
    // I8: concat
    {
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f}},
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f}},
            ngraph::element::i8,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::i8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, { 0.005f } }
        }
    },
    // U8: concat with subtract
    {
        LayerTransformation::createParamsU8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {1.275f}, {2.55f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {0.f}, {255.f} },
            ngraph::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::u8,
            {
                ngraph::element::f32,
                {{ 0.f, 0.f, 0.f, -255.f, -255.f, -255.f }},
                {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }}
            },
            { ngraph::element::f32, {-255.f}, { 0.005f } }
        }
    },
    // U8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}},
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, { 255.f}},
            ngraph::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, { 0.005f } }
        }
    },
    // U8: concat multi channels with per-channel quantization
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} },
            {
                256ul,
                ngraph::Shape({ 1, 6, 1, 1 }),
                {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                {255.f, 25.5f, 2.55f, 25.5f, 255.f, 2.55f},
                {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                {255.f, 25.5f, 2.55f, 25.5f, 255.f, 2.55f}
            }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {255.f}},
            {
                256ul,
                ngraph::Shape({ 1, 6, 1, 1 }),
                {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                {255.f, 25.5f, 2.55f, 25.5f, 255.f, 2.55f},
                {0.f},
                {255.f}
            },
            ngraph::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 1.f, 0.1f, 0.01f }} },
            { ngraph::element::f32, {}, {{ 0.1f, 1.f, 0.01f }} }
        }
    },
    // I8: concat multi channels
    {
        LayerTransformation::createParamsI8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f}},
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f}},
            ngraph::element::i8,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::i8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, { 0.005f } }
        }
    },
    // not update precisions
    {
        LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
        true,
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f}},
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f}},
            ngraph::element::f32,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::f32,
            { {}, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { {}, {}, { 0.005f } }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithSplitTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues),
        ::testing::Values(true)),
    ConcatWithSplitTransformation::getTestCaseName);
} // namespace casesWithConvolution

// test cases to check output names
namespace casesWithoutConvolution {
const std::vector<ConcatTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}},
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, { 255.f}},
            ngraph::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, { 0.005f } }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithSplitTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues),
        ::testing::Values(false)),
    ConcatWithSplitTransformation::getTestCaseName);
} // namespace casesWithoutConvolution
} // namespace
