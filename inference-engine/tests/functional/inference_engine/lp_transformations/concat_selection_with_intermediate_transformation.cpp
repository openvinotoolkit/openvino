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
#include <low_precision/max_pool.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
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
    ngraph::element::Type precisionBeforeOp;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore1;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore2;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter1;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter2;
};

inline std::ostream& operator<<(std::ostream& out, const ResultValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_"
               << values.dequantizationAfter1 << "_" << values.dequantizationAfter2;
}

class TestValues {
public:
    ngraph::Shape inputShape;
    TestTransformationParams params;
    bool transparentIntermediate;
    ActualValues actual;
    ResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const TestValues& values) {
    return out << "_" << values.transparentIntermediate << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    TestValues
> ConcatTransformationParams;

class ConcatSelectionWithIntermediateTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        TestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginalSelectionWithIntermediate(
            precision,
            testValues.inputShape,
            testValues.transparentIntermediate,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2);

        auto supportedPrecisions = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
            ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}}
            })
        });

        SimpleLowPrecisionTransformer transform(supportedPrecisions);
        transform.add<ngraph::pass::low_precision::ConcatTransformation, ngraph::opset1::Concat>(testValues.params);
        transform.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReferenceSelectionWithIntermediate(
            precision,
            testValues.inputShape,
            testValues.transparentIntermediate,
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
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const TestValues testValues = std::get<1>(obj.param);

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
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<TestValues> testValues = {
    // U8: Concat + MaxPool
    {
        Shape{ 1, 3, 9, 9 },
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {25.5f}, {0.f}, {255.f} },
            ngraph::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::u8,
            { {ngraph::element::f32}, {}, { {0.01f, 0.01f, 0.01f, 0.1f, 0.1f, 0.1f} } },
            { {ngraph::element::f32}, {}, { 0.1f } }
        }
    },
    // not update precisions
    {
        Shape{ 1, 3, 9, 9 },
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        true,
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, ngraph::Shape({}), {0.f}, {25.5f}, {0.f}, {255.f} },
            ngraph::element::f32,
            {{}, {}, {}},
            {{}, {}, {}},
            ngraph::element::f32,
            { {}, {}, { {0.01f, 0.01f, 0.01f, 0.1f, 0.1f, 0.1f} } },
            { {}, {}, { 0.1f } }
        }
    }
};

 INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatSelectionWithIntermediateTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(testValues)),
    ConcatSelectionWithIntermediateTransformation::getTestCaseName);
}  // namespace
