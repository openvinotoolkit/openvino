// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "low_precision/prelu.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/max_pool.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/fake_quantize_precision_selection.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

namespace {
class ActualValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

class ExpectedValues {
public:
    element::Type fakeQuantizeOnDataOutPrecision;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

class FakeQuantizePrecisionSelectionTransformationTestValues {
public:
    std::vector<element::Type> precisionsOnActivations;
    std::vector<element::Type> precisionsOnActivationForLimitedOperation;
    bool operationBeforeLimitedOperationIsPrecisionTransparent;
    ActualValues actual;
    ExpectedValues expected;
};

inline std::ostream& operator<<(std::ostream& out, const ActualValues& values) {
    return out << values.fakeQuantizeOnData << "_" << values.fakeQuantizeOnWeights;
}

inline std::ostream& operator<<(std::ostream& out, const ExpectedValues& values) {
    return out << values.fakeQuantizeOnDataOutPrecision << "_" << values.fakeQuantizeOnData << "_" << values.fakeQuantizeOnWeights;
}

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizePrecisionSelectionTransformationTestValues& testValue) {
    return out << "_" << testValue.precisionsOnActivationForLimitedOperation[0] << "_" << testValue.actual << "_" << testValue.expected;
}

typedef std::tuple<
    ov::element::Type,
    ov::Shape,
    bool,
    FakeQuantizePrecisionSelectionTransformationTestValues> FakeQuantizePrecisionSelectionTransformationParams;

class FakeQuantizePrecisionSelectionTransformation : public LayerTransformation,
    public testing::WithParamInterface<FakeQuantizePrecisionSelectionTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::Shape shape = std::get<1>(GetParam());
        const bool updatePrecision = std::get<2>(GetParam());
        const FakeQuantizePrecisionSelectionTransformationTestValues testValues = std::get<3>(GetParam());

        auto params = createParamsU8I8AndI8();
        params.setUpdatePrecisions(updatePrecision);
        params.setPrecisionsOnActivations(testValues.precisionsOnActivations);

        auto precisionLimitedOperationParams(params);
        precisionLimitedOperationParams.setPrecisionsOnActivations(testValues.precisionsOnActivationForLimitedOperation);

        actualFunction = ov::builder::subgraph::FakeQuantizePrecisionSelectionFunction::getOriginal(
            precision,
            shape,
            {
                testValues.operationBeforeLimitedOperationIsPrecisionTransparent,
                testValues.actual.fakeQuantizeOnData,
                testValues.actual.fakeQuantizeOnWeights
            });

        auto supportedPrecisions = std::vector<ov::pass::low_precision::PrecisionsRestriction>({
           ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::Convolution>({
               {{0}, testValues.precisionsOnActivationForLimitedOperation},
               {{1}, { element::i8 }}
           })
        });

        SimpleLowPrecisionTransformer transform(supportedPrecisions);
        transform.add<ov::pass::low_precision::PReluTransformation, ov::op::v0::PRelu>(params);
        transform.add<ov::pass::low_precision::ConvolutionTransformation, ov::op::v1::Convolution>(precisionLimitedOperationParams);
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(params);
        transform.add<ov::pass::low_precision::MaxPoolTransformation, ov::op::v1::MaxPool>(params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::FakeQuantizePrecisionSelectionFunction::getReference(
            precision,
            shape,
            {
                testValues.operationBeforeLimitedOperationIsPrecisionTransparent,
                updatePrecision ? testValues.expected.fakeQuantizeOnDataOutPrecision : precision,
                testValues.expected.fakeQuantizeOnData,
                testValues.expected.fakeQuantizeOnWeights
            });
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizePrecisionSelectionTransformationParams> obj) {
        ov::element::Type precision;
        ov::Shape shape;
        bool updatePrecision;
        FakeQuantizePrecisionSelectionTransformationTestValues testValues;
        std::tie(precision, shape, updatePrecision, testValues) = obj.param;

        TestTransformationParams params;
        params.setUpdatePrecisions(updatePrecision);
        params.setPrecisionsOnActivations(testValues.precisionsOnActivations);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, params) << testValues;
        return result.str();
    }
};

TEST_P(FakeQuantizePrecisionSelectionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    // ov::element::f16
};

const std::vector<bool> updatePrecisions = {
    true,
    false
};

const std::vector<FakeQuantizePrecisionSelectionTransformationTestValues> fakeQuantizeTransformationTestValues = {
    {
        { element::u8, element::i8 },
        { element::u8 },
        true,
        {
            { 256ul, { }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        {
            element::u8,
            { 256ul, { }, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
            { }
        },
    },
    {
        { element::u8, element::i8 },
        { element::i8 },
        true,
        {
            { 256ul, { }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        {
            { element::i8 },
            { 256ul, { }, { -1.28f }, { 1.27f }, { -128.f }, { 127.f } },
            { }
        },
    },
    // {
    //    { element::u8, element::i8 },
    //    { element::i8 },
    //    // INT8 is not available for limited operation (Convolution)
    //    false,
    //    {
    //        { 256ul, { }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
    //        { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
    //    },
    //    {
    //        // original precision is used
    //        element::u8,
    //        // FakeQuantize has to select the first available: U8, not limited operation required I8 but this fact doesn't affect
    //        { 256ul, { }, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
    //        // FakeQuantize on weights is not changed
    //        { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
    //    },
    // },
};

const std::vector<ov::Shape> shapes = {
    { 1, 32, 72, 48 },
    // TODO: 3D tensor
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FakeQuantizePrecisionSelectionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(fakeQuantizeTransformationTestValues)),
    FakeQuantizePrecisionSelectionTransformation::getTestCaseName);

} // namespace
