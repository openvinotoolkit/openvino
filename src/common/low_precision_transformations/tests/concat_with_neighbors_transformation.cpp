// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"

#include "low_precision/align_quantization_intervals.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/markup_precisions.hpp"
#include "low_precision/markup_avg_pool_precision_preserved.hpp"
#include "low_precision/propagate_precisions.hpp"

#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/concat.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

namespace {

class ConcatTransformationActualValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.fakeQuantize3;
}

class ConcatTransformationResultValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
    ov::element::Type precisionBeforeOp;
    ov::builder::subgraph::DequantizationOperations dequantizationBefore;
    ov::element::Type precisionAfterOp;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter1;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter2;
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
    TestTransformationParams params;
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
    ov::element::Type,
    ov::PartialShape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatWithNeighborsTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::PartialShape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ov::builder::subgraph::ConcatFunction::getOriginalWithNeighbors(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3,
            testValues.neighborType,
            testValues.additionalLayer);

        auto supportedPrecisionsOnActivation = std::vector<ov::pass::low_precision::PrecisionsRestriction>({
            ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::Convolution>({
                {{0}, testValues.params.precisionsOnActivations},
                {{1}, testValues.params.precisionsOnWeights}
            })
        });

        auto quantizationRestrictions = testValues.multiChannels ?
            std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>() :
            std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>({
                ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::op::v1::Convolution>()
        });

        SimpleLowPrecisionTransformer transform(supportedPrecisionsOnActivation, quantizationRestrictions);
        transform.add<ov::pass::low_precision::ConcatTransformation, ov::op::v0::Concat>(testValues.params);
        transform.add<ov::pass::low_precision::ConvolutionTransformation, ov::op::v1::Convolution>(testValues.params);
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::ConcatFunction::getReferenceWithNeighbors(
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
        const ov::element::Type precision = std::get<0>(obj.param);
        const ov::PartialShape shape = std::get<1>(obj.param);
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    // ov::element::f16
};

const std::vector<ov::PartialShape> shapes = {
    { 1, 3, 9, 9 },
    { 4, 3, 9, 9 },
    { Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() }
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            { ov::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ov::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
        },
        "concat",
        ""
    },
    // U8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            { ov::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ov::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
        },
        "concat",
        ""
    },
    // U8: concat multi channels with subtract
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {1.275f}, {2.55f}, {1.275f}, {2.55f} },
            { 256ul, ov::Shape({}), {1.275f}, {2.55f}, {1.275f}, {2.55f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {1.275f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {1.275f}, {2.55f}, {0.f}, {255.f} },
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            { ov::element::f32, {{ 0.f, 0.f, 0.f, -255.f, -255.f, -255.f }}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ov::element::f32, { -255.f }, { 0.005f } }
        },
        "concat",
        ""
    },
    // I8: concat
    {
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ov::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} }
        },
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-128.f}, {127.f} },
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            { ov::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ov::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
        },
        "concat",
        ""
    },
    // I8: concat multi channels
    {
        LayerTransformation::createParamsI8I8(),
        true,
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ov::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} }
        },
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-128.f}, {127.f} },
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            { ov::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { ov::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
        },
        "concat",
        ""
    },
    // mixed: U8 + I8: concat multi channels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            { ov::element::f32, {{ -128.f, -128.f, -128.f, 0.f, 0.f, 0.f }}, { 0.01f } },
            { ov::element::f32, {}, { 0.01f } }
        },
        "concat",
        ""
    },
    // not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            ov::element::f32,
            {{}, {}, {}},
            ov::element::f32,
            { {}, {{ -128.f, -128.f, -128.f, 0.f, 0.f, 0.f }}, { 0.01f } },
            { {}, {}, { 0.01f } }
        },
        "concat",
        ""
    },
    // convolution neighbor and additional layer
    // different precisions on FQ, u8 have to be chosen
    {
        LayerTransformation::createParamsU8I8(),
        false,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-12.8f}, {12.7f} },
            {}
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {128.f}, {154.f} },
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            {},
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {
                {},
                {{ 128.f, 128.f, 128.f, 128.f, 128.f, 128.f }, ov::element::f32, { 1, 6, 1, 1 }, false},
                {{0.1f}, ov::element::f32, {} } },
            {
                {},
                {{128.f, 128.f, 128.f}, ov::element::f32, { 1, 3, 1, 1 }, false},
                {{0.1f}, ov::element::f32, {} } }
        },
        "convolution",
        "convolution"
    },
    //// I8: concat multi channels
    //{
    //    LayerTransformation::createParamsI8I8(),
    //    true,
    //    {
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
    //        { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
    //        { 256ul, ov::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} }
    //    },
    //    {
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
    //        { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f} },
    //        { 256ul, ov::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-128.f}, {127.f} },
    //        ov::element::i8,
    //        {{}, {}, {}},
    //        ov::element::i8,
    //        { ov::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
    //        { ov::element::f32, {}, {{ 0.005f, 0.005f, 0.005f, 0.00333f, 0.00333f, 0.00333f }} }
    //    }
    //},
    //// mixed: U8 + I8: concat multi channels
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    true,
    //    {
    //        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
    //    },
    //    {
    //        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
    //        ov::element::u8,
    //        {{}, {}, {}},
    //        ov::element::u8,
    //        { ov::element::f32, {{ 0.f, 0.f, 0.f, 128.f, 128.f, 128.f }}, { 0.01f } },
    //        { ov::element::f32, { 128.f }, { 0.01f } }
    //    }
    //},
    //// not update precisions
    //{
    //    LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
    //    true,
    //    {
    //        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
    //    },
    //    {
    //        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
    //        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
    //        ov::element::f32,
    //        {{}, {}, {}},
    //        ov::element::f32,
    //        { {}, {{ 0.f, 0.f, 0.f, 128.f, 128.f, 128.f }}, { 0.01f } },
    //        { {}, { 128.f }, { 0.01f } }
    //    }
    //},
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
