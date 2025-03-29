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
#include "low_precision/concat.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/strided_slice.hpp"

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
    ov::builder::subgraph::DequantizationOperations dequantizationBefore;
    ov::element::Type precisionBeforeConcat;
    ov::element::Type precisionAfterConcat;
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
    bool ssBeforeConcat;
    bool ssAfterConcat;
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

class ConcatWithStridedSliceTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::PartialShape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ov::builder::subgraph::ConcatFunction::getOriginalWithStridedSlice(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.ssBeforeConcat,
            testValues.ssAfterConcat);

        auto supportedPrecisions = std::vector<ov::pass::low_precision::PrecisionsRestriction>({
           ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::Convolution>({
               {{0}, testValues.params.precisionsOnActivations},
               {{1}, testValues.params.precisionsOnWeights},
           })
       });

        auto quantizationRestrictions = testValues.multiChannels ?
            std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>() :
            std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>({
                ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::op::v1::Convolution>()
            });

        SimpleLowPrecisionTransformer transform(supportedPrecisions, quantizationRestrictions);
        transform.add<ov::pass::low_precision::ConcatTransformation, ov::op::v0::Concat>(testValues.params);
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(testValues.params);
        transform.add<ov::pass::low_precision::MaxPoolTransformation, ov::op::v1::MaxPool>(testValues.params);
        transform.add<ov::pass::low_precision::StridedSliceTransformation, ov::op::v1::StridedSlice>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::ConcatFunction::getReferenceWithStridedSlice(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.dequantizationBefore,
            testValues.result.precisionBeforeConcat,
            testValues.result.precisionAfterConcat,
            testValues.ssBeforeConcat,
            testValues.ssAfterConcat,
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
            (testValues.ssBeforeConcat ? "SS_before_concat_" : "") <<
            (testValues.ssAfterConcat ? "SS_after_cancat_" : "") <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithStridedSliceTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    // ov::element::f16
};

const std::vector<ov::PartialShape> shapes = {
    { 1, 4, 9, 9 },
    { 4, 4, 9, 9 },
    { Dimension::dynamic(), 4, Dimension::dynamic(), Dimension::dynamic() }
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // FQ with the same values, ss before concat, ss after concat
    {
        LayerTransformation::createParamsU8I8(),
        true,
        true,
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            {ov::element::f32, {}, { 0.01f }},
            ov::element::u8,
            ov::element::u8,
            {ov::element::f32, {}, { 0.01f }},
            {ov::element::f32, {}, { 0.01f }}
        }
    },
    // FQ with different values, ss before concat, ss after concat
    {
        LayerTransformation::createParamsU8I8(),
        true,
        true,
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {255.f} },
            {ov::element::f32, {}, { 0.01f }},
            ov::element::u8,
            ov::element::u8,
            {ov::element::f32, {}, { {0.01f, 0.01f, 0.1f, 0.1f} }},
            {ov::element::f32, {}, { {0.01f, 0.01f, 0.1f, 0.1f, 0.1f, 0.1f} }}
        }
    },
    // FQ with different values, ss after concat
    {
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        true,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {255.f} },
            {ov::element::f32, {}, { 0.01f }},
            ov::element::u8,
            ov::element::u8,
            {ov::element::f32, {}, { {0.01f, 0.01f, 0.01f, 0.01f, 0.1f, 0.1f} }},
            {ov::element::f32, {}, { {0.01f, 0.01f, 0.01f, 0.01f, 0.1f, 0.1f, 0.1f, 0.1f} }}
        }
    },
    // FQ with different values, ss before concat
    {
        LayerTransformation::createParamsU8I8(),
        true,
        true,
        false,
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {255.f} },
            {ov::element::f32, {}, { 0.01f }},
            ov::element::u8,
            ov::element::u8,
            {ov::element::f32, {}, { {0.01f, 0.01f, 0.1f, 0.1f, 0.1f, 0.1f} }},
            {ov::element::f32, {}, { {0.01f, 0.01f, 0.1f, 0.1f, 0.1f, 0.1f} }}
        }
    },
    // FQ with zero-point, ss before concat, ss after concat
    {
        LayerTransformation::createParamsU8I8(),
        true,
        true,
        true,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f} }
        },
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, {}, {1.275f}, {2.55f}, {0.f}, {255.f} },
            {ov::element::f32, {}, { 0.01f }},
            ov::element::u8,
            ov::element::u8,
            {ov::element::f32, { {0.f, 0.f, -255.f, -255.f} }, { {0.01f, 0.01f, 0.005f, 0.005f} }},
            {ov::element::f32, { {0.f, 0.f, -255.f, -255.f, -255.f, -255.f} }, { {0.01f, 0.01f, 0.005f, 0.005f, 0.005f, 0.005f} }}
        }
    },
    // not multi channels concat, ss before concat, ss after concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        true,
        true,
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
        },
        {
            { 256ul, {}, {0.f}, {2.55f}, {85.f}, {255.f} },
            { 256ul, {}, {-1.28f}, {1.27f}, {0.f}, {170.f} },
            {ov::element::f32, { 85 }, { 0.015f } },
            ov::element::u8,
            ov::element::u8,
            {ov::element::f32, { 85 }, { 0.015f } },
            {ov::element::f32, { 85 }, { 0.015f } }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithStridedSliceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatWithStridedSliceTransformation::getTestCaseName);
}  // namespace
