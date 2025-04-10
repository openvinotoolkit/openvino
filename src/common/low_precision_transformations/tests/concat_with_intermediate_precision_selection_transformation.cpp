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
#include "low_precision/avg_pool.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/max_pool.hpp"
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
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2;
}

class ConcatTransformationResultValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ov::element::Type precisionBeforeOp;
    ov::builder::subgraph::DequantizationOperations dequantizationBefore1;
    ov::builder::subgraph::DequantizationOperations dequantizationBefore2;
    ov::element::Type precisionAfterOperation;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter1;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.dequantizationAfter1 << "_" <<
        values.dequantizationAfter2;
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
    ov::element::Type,
    ov::PartialShape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatWithIntermediatePrecisionSelectionTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::PartialShape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ov::builder::subgraph::ConcatFunction::getOriginalWithIntermediateAvgPool(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2);

        auto supportedPrecisionsOnActivation = std::vector<ov::pass::low_precision::PrecisionsRestriction>({
            ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::AvgPool>({{{0}, testValues.params.precisionsOnActivations}})
        });

        auto quantizationRestrictions = testValues.multiChannels ?
            std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>() :
            std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>({
                ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::op::v1::AvgPool>()
            });

        SimpleLowPrecisionTransformer transform(supportedPrecisionsOnActivation, quantizationRestrictions);
        transform.add<ov::pass::low_precision::ConcatTransformation, ov::op::v0::Concat>(testValues.params);
        transform.add<ov::pass::low_precision::MaxPoolTransformation, ov::op::v1::MaxPool>(testValues.params);
        transform.add<ov::pass::low_precision::AvgPoolTransformation, ov::op::v1::AvgPool>(testValues.params);
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::ConcatFunction::getReferenceWithIntermediateAvgPool(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.precisionBeforeOp,
            testValues.result.dequantizationBefore1,
            testValues.result.dequantizationBefore2,
            testValues.result.precisionAfterOperation,
            testValues.result.dequantizationAfter1,
            testValues.result.dequantizationAfter2);
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

TEST_P(ConcatWithIntermediatePrecisionSelectionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, false, false);
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
    // Concat: FakeQuantize operations with signed intervals but consumer requires U8
    {
        LayerTransformation::createParamsU8I8(),
        false,
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} }
        },
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {64.f}, {192.f} },
            ov::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ov::element::u8,
            { ov::element::f32, { 128.f }, { 0.01f } },
            { {}, { 128.f }, { 0.01f } }
        }
    },

    // Concat: FakeQuantize operations with unsigned intervals but consumer requires I8
    {
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f / 2.f}, {-128.f}, { -0.f} },
            ov::element::i8,
            {{}, {}, {}},
            {{}, {}, {}},
            ov::element::i8,
            { ov::element::f32, { -128.f }, { 0.01f } },
            { {}, { -128.f }, { 0.01f } }
        }
    },

    // ConcatMultichannel: FakeQuantize operations with signed intervals but consumer requires U8
    {
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} }
        },
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {0.f}, { 255.f} },
            ov::element::u8,
            {},
            {},
            ov::element::u8,
            { ov::element::f32, { 128.f }, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { {}, { 128.f }, { 0.005f } }
        }
    },

    // ConcatMultichannel: FakeQuantize operations with unsigned intervals but consumer requires I8
    {
        LayerTransformation::createParamsI8I8(),
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f / 2.f}, {-128.f}, { 127.f} },
            ov::element::i8,
            {{}, {}, {}},
            {{}, {}, {}},
            ov::element::i8,
            { ov::element::f32, { -128.f }, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { {}, { -128.f }, { 0.005f } }
        }
    },

    // Concat: FakeQuantize operations with unsigned intervals, no consumer limitations: FQ were decomposed to U8 precision
    {
        LayerTransformation::createParamsU8I8AndI8(),
        false,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, { 128.f} },
            ov::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ov::element::u8,
            { ov::element::f32, {}, { 0.01f } },
            { {}, {}, { 0.01f } }
        }
    },

    // Concat: FakeQuantize operations with signed intervals, no consumer limitations: FQ were decomposed to I8 precision
    {
        LayerTransformation::createParamsU8I8AndI8(),
        false,
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} }
        },
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-64.f}, {64.f} },
            ov::element::i8,
            {{}, {}, {}},
            {{}, {}, {}},
            ov::element::i8,
            { ov::element::f32, {}, { 0.01f } },
            { {}, {}, { 0.01f } }
        }
    },

    // ConcatMultichannel: FakeQuantize operations with unsigned intervals, no consumer limitations: FQ were decomposed to U8 precision
    {
        LayerTransformation::createParamsU8I8AndI8(),
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {255.f} },
            ov::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ov::element::u8,
            { ov::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { {}, {}, { 0.005f } }
        }
    },

    // ConcatMultichannel: FakeQuantize operations with signed intervals, no consumer limitations: FQ were decomposed to I8 precision
    {
        LayerTransformation::createParamsU8I8AndI8(),
        true,
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} }
        },
        {
            { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ov::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-128.f}, {127.f} },
            ov::element::i8,
            {{}, {}, {}},
            {{}, {}, {}},
            ov::element::i8,
            { ov::element::f32, {}, {{ 0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f }} },
            { {}, {}, { 0.005f } }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithIntermediatePrecisionSelectionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatWithIntermediatePrecisionSelectionTransformation::getTestCaseName);
}  // namespace
