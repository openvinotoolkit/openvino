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
#include <low_precision/transformer.hpp>
#include <low_precision/concat.hpp>
#include <low_precision/concat_multi_channels.hpp>
#include <low_precision/convolution.hpp>

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
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.fakeQuantize3;
}

class ConcatTransformationResultValues {
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

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.fakeQuantize3 << "_" <<
        values.dequantizationAfter1 << "_" <<
        values.dequantizationAfter2;
}

class ConcatTransformationTestValues {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannels;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
    std::string neighborType;
    std::string additionalLayer;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::Shape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatWithNeighborsTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginalWithNeighbors(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3,
            testValues.neighborType,
            testValues.additionalLayer);

        SimpleLowPrecisionTransformer transformBranchSpecific;
        if (testValues.multiChannels) {
            transformBranchSpecific.add<ngraph::pass::low_precision::ConcatMultiChannelsTransformation, ngraph::opset1::Concat>(testValues.params);
        } else {
            transformBranchSpecific.add<ngraph::pass::low_precision::ConcatTransformation, ngraph::opset1::Concat>(testValues.params);
        }
        if (testValues.additionalLayer == "convolution" || testValues.neighborType == "convolution") {
            transformBranchSpecific.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        }
        transformBranchSpecific.transform(actualFunction);
        if (testValues.additionalLayer == "convolution" || testValues.neighborType == "convolution") {
            SimpleLowPrecisionTransformer transformConvolution;
            transformConvolution.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
            transformConvolution.transform(actualFunction);
        }

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReferenceWithNeighbors(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.fakeQuantize3,
            testValues.result.precisionBeforeOp,
            testValues.result.dequantizationBefore,
            testValues.result.precisionAfterOp,
            testValues.result.dequantizationAfter1,
            testValues.result.dequantizationAfter2,
            testValues.neighborType,
            testValues.additionalLayer);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithNeighborsTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {128.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {85.f} },
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } },
            { ngraph::element::f32, {}, { 0.01f } }
        },
        "concat",
        ""
    },
    // U8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
        },
        "concat",
        ""
    },
    // U8: concat multi channels with subtract
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {1.275f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {1.275f}, {2.55f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {1.275f}, {2.55f}, {0.f}, {255.f} },
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {{ 0.f, 0.f, 0.f, -255.f, -255.f, -255.f }}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, { -255.f }, { 0.005f } }
        },
        "concat",
        ""
    },
    // I8: concat
    {
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-64}, {64.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-43}, {42.f} },
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.01f } },
            { ngraph::element::f32, {}, { 0.01f } }
        },
        "concat",
        ""
    },
    // I8: concat multi channels
    {
        LayerTransformation::createParamsI8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-128.f}, {127.f} },
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ngraph::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
        },
        "concat",
        ""
    },
    // mixed: U8 + I8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            { ngraph::element::f32, {{ 0.f, 0.f, 0.f, 128.f, 128.f, 128.f }}, { 0.01f } },
            { ngraph::element::f32, { 128.f }, { 0.01f } }
        },
        "concat",
        ""
    },
    // not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            ngraph::element::f32,
            {{}, {}, {}},
            ngraph::element::f32,
            { {}, {{ 0.f, 0.f, 0.f, 128.f, 128.f, 128.f }}, { 0.01f } },
            { {}, { 128.f }, { 0.01f } }
        },
        "concat",
        ""
    },
    // convolution neighbor and additional layer
    // different precisions on FQ, u8 have to be chosen
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-12.8f}, {12.7f} },
            {}
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {128.f}, {154.f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            {},
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {},
                {{ 128.f, 128.f, 128.f, 128.f, 128.f, 128.f }, ngraph::element::f32, { 1, 6, 1, 1 }, false},
                {{0.1f}, ngraph::element::f32, { 1, 1, 1, 1 } } },
            {
                {},
                {{128.f, 128.f, 128.f}, ngraph::element::f32, { 1, 3, 1, 1 }, false},
                {{0.1f}, ngraph::element::f32, { 1, 1, 1, 1 } } }
        },
        "convolution",
        "convolution"
    },
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 },
    { 4, 3, 9, 9 }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithNeighborsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatWithNeighborsTransformation::getTestCaseName);
}  // namespace
