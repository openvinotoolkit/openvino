// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/low_precision/avg_pool.hpp>
#include <transformations/low_precision/convolution.hpp>
#include <transformations/low_precision/fake_quantize.hpp>
#include <transformations/low_precision/max_pool.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/fake_quantize_precision_selection_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {
class ActualValues {
public:
    builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

class ExpectedValues {
public:
    element::Type fakeQuantizeOnDataOutPrecision;
    builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

class FakeQuantizePrecisionSelectionTransformationTestValues {
public:
    std::vector<element::Type> precisionsOnActivations;
    std::vector<element::Type> precisionsOnActivationForLimitedOperation;
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
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    FakeQuantizePrecisionSelectionTransformationTestValues> FakeQuantizePrecisionSelectionTransformationParams;

class FakeQuantizePrecisionSelectionTransformation : public LayerTransformation, public testing::WithParamInterface<FakeQuantizePrecisionSelectionTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const bool updatePrecision = std::get<2>(GetParam());
        const FakeQuantizePrecisionSelectionTransformationTestValues testValues = std::get<3>(GetParam());

        low_precision::LayerTransformation::Params params = createParamsU8I8AndI8();
        params.setUpdatePrecisions(updatePrecision);
        params.setPrecisionsOnActivations(testValues.precisionsOnActivations);

        low_precision::LayerTransformation::Params precisionLimitedOperationParams(params);
        precisionLimitedOperationParams.setPrecisionsOnActivations(testValues.precisionsOnActivationForLimitedOperation);

        actualFunction = ngraph::builder::subgraph::FakeQuantizePrecisionSelectionFunction::getOriginal(
            precision,
            shape,
            { testValues.actual.fakeQuantizeOnData, testValues.actual.fakeQuantizeOnWeights });

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AvgPoolTransformation, ngraph::opset1::AvgPool>(params);
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(precisionLimitedOperationParams);
        transform.add<ngraph::pass::low_precision::FakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FakeQuantizePrecisionSelectionFunction::getReference(
            precision,
            shape,
            {
                updatePrecision ? testValues.expected.fakeQuantizeOnDataOutPrecision : precision,
                testValues.expected.fakeQuantizeOnData,
                testValues.expected.fakeQuantizeOnWeights
            });
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizePrecisionSelectionTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool updatePrecision;
        FakeQuantizePrecisionSelectionTransformationTestValues testValues;
        std::tie(precision, shape, updatePrecision, testValues) = obj.param;

        low_precision::LayerTransformation::Params params;
        params.setUpdatePrecisions(updatePrecision);
        params.setPrecisionsOnActivations(testValues.precisionsOnActivations);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, params) << testValues;
        return result.str();
    }
};

TEST_P(FakeQuantizePrecisionSelectionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<bool> updatePrecisions = {
    true,
    false
};

const std::vector<FakeQuantizePrecisionSelectionTransformationTestValues> fakeQuantizeTransformationTestValues = {
    // U8
    {
        { element::u8, element::i8 },
        { element::u8 },
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
        {
            { 256ul, { }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
            { 255ul, { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        {
            { element::i8 },
            { 256ul, { }, { -1.28f }, { 1.27f }, { -128.f }, { 127.f } },
            { }
        },
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 },
    // TODO: 3D tensor
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    FakeQuantizePrecisionSelectionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(fakeQuantizeTransformationTestValues)),
    FakeQuantizePrecisionSelectionTransformation::getTestCaseName);

}
