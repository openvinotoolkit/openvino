// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/concat.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/concat.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

namespace {
class ActualValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
};

inline std::ostream& operator<<(std::ostream& out, const ActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2;
}

class ResultValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const ResultValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_";
}

class TestValues {
public:
    ov::Shape inputShape;
    ov::Shape reshapeOutputShape;
    TestTransformationParams params;
    ActualValues actual;
    ResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const TestValues& values) {
    return out << "_" << values.reshapeOutputShape << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
        ov::element::Type,
        TestValues
> ConcatTransformationParams;

class ConcatWithIntermediateReshapeTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        TestValues testValues = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::ConcatFunction::getOriginalWithIntermediateReshape(
            precision,
            testValues.inputShape,
            testValues.reshapeOutputShape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::ConcatTransformation, ov::op::v0::Concat>(testValues.params);
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(testValues.params);
        transform.add<ov::pass::low_precision::ReshapeTransformation, ov::op::v1::Reshape>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::ConcatFunction::getReferenceWithIntermediateReshape(
            precision,
            testValues.inputShape,
            testValues.reshapeOutputShape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ov::element::Type precision = std::get<0>(obj.param);
        const TestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
               LayerTransformation::getTestCaseNameByParams(precision, testValues.inputShape, testValues.params) << "_" <<
               testValues.reshapeOutputShape << "_" <<
               testValues.actual << "_" <<
               testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithIntermediateReshapeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {
        ov::element::f32,
        // ov::element::f16
};

const std::vector<TestValues> testValues = {
    // U8: Concat + MaxPool
    {
        Shape{ 2, 1, 9 },
        Shape{ 2, 1, 1, 9 },
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} }
        },
        {
            { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {255.f} },
            { {ov::element::f32}, {}, { {0.01f, 0.1f} } }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithIntermediateReshapeTransformation,
    ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(testValues)),
    ConcatWithIntermediateReshapeTransformation::getTestCaseName);
}  // namespace
