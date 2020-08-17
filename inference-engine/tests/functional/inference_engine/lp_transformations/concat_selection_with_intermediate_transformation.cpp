// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/concat.hpp>
#include <transformations/low_precision/concat_multi_channels.hpp>
#include <transformations/low_precision/max_pool.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class ActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
};

inline std::ostream& operator<<(std::ostream& out, const ActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2;
}

class ResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
};

inline std::ostream& operator<<(std::ostream& out, const ResultValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.dequantization1 << "_" << values.dequantization2;
}

class TestValues {
public:
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool transparentIntermediate;
    ActualValues actual;
    ResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const TestValues& values) {
    return out << "_" << values.transparentIntermediate << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    bool,
    TestValues
> ConcatTransformationParams;

class ConcatSelectionWithIntermediateTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const bool updatePrecisions = std::get<1>(GetParam());
        TestValues testValues = std::get<2>(GetParam());

        testValues.params.updatePrecisions = updatePrecisions;
        if (!updatePrecisions) {
            testValues.result.fakeQuantize1.outputPrecision = testValues.actual.fakeQuantize1.outputPrecision;
            testValues.result.fakeQuantize2.outputPrecision = testValues.actual.fakeQuantize2.outputPrecision;
        }

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginalSelectionWithIntermediate(
            precision,
            testValues.inputShape,
            testValues.transparentIntermediate,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConcatMultiChannelsTransformation, ngraph::opset1::Concat>(testValues.params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReferenceSelectionWithIntermediate(
            precision,
            testValues.inputShape,
            testValues.transparentIntermediate,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.dequantization1,
            testValues.result.dequantization2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const bool updatePrecision = std::get<1>(obj.param);
        const TestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, testValues.inputShape, testValues.params) << "_" <<
            testValues.transparentIntermediate << "_" <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatSelectionWithIntermediateTransformation, CompareFunctions) {
    const TestValues testValues = std::get<2>(GetParam());

    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<bool> updatePrecisions = { true, false };

const std::vector<TestValues> testValues = {
    // U8: Concat + MaxPool
    {
        Shape{ 1, 3, 9, 9 },
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f / 2.f}, {0.f}, { 128.f}, ngraph::element::u8 },
            { ngraph::element::f32, {}, { 0.01f } },
            { ngraph::element::f32, {}, { 0.01f } }
        }
    }
};

// INSTANTIATE_TEST_CASE_P(
//    DISABLED_LPT,
//    ConcatSelectionWithIntermediateTransformation,
//    ::testing::Combine(
//        ::testing::ValuesIn(precisions),
//        ::testing::ValuesIn(updatePrecisions),
//        ::testing::ValuesIn(testValues)),
//    ConcatSelectionWithIntermediateTransformation::getTestCaseName);
}  // namespace
