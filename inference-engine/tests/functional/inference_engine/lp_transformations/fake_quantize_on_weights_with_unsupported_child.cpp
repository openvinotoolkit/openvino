// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/fake_quantize_on_weights_and_unsupported_child_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FakeQuantizeOnWeightsWithUnsupportedChildTestValues {
public:
    class Actual {
    public:
        std::shared_ptr<ngraph::opset1::Constant> weights;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class Expected {
    public:
        std::shared_ptr<ngraph::opset1::Constant> weights;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::element::Type precision;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::Shape,
    FakeQuantizeOnWeightsWithUnsupportedChildTestValues> FakeQuantizeOnWeightsWithUnsupportedChildParams;

class FakeQuantizeOnWeightsWithUnsupportedChild :
    public LayerTransformation,
    public testing::WithParamInterface<FakeQuantizeOnWeightsWithUnsupportedChildParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::FakeQuantizeOnWeightsAndUnsupportedChildFunction::get(
            inputShape,
            testValues.precision,
            testValues.actual.weights,
            testValues.actual.fakeQuantizeOnWeights);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeOnWeightsAndUnsupportedChildFunction::get(
            inputShape,
            testValues.precision,
            testValues.expected.weights,
            testValues.expected.fakeQuantizeOnWeights);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeOnWeightsWithUnsupportedChildParams> obj) {
        auto inputShape = std::get<0>(obj.param);
        FakeQuantizeOnWeightsWithUnsupportedChildTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
            inputShape << "_" << testValues.precision << testValues.actual.fakeQuantizeOnWeights;
        return result.str();
    }
};

TEST_P(FakeQuantizeOnWeightsWithUnsupportedChild, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> shapes = {
    ngraph::Shape({ 1, 3, 72, 48 }),
    ngraph::Shape({ 4, 3, 72, 48 })
};

const std::vector<FakeQuantizeOnWeightsWithUnsupportedChildTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        ngraph::element::f32,
        {
            op::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1 }, std::vector<float>{ 1.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -127.f }, { 127.f } }
        },
        {
            op::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1 }, std::vector<float>{ -126.f }),
            {},
        }
    },
    {
        LayerTransformation::createParamsU8U8(),
        ngraph::element::f32,
        {
            op::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1 }, std::vector<float>{ 1.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { 0.f }, { 254.f } }
        },
        {
            op::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1 }, std::vector<float>{ 1.f }),
            {},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FakeQuantizeOnWeightsWithUnsupportedChild,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    FakeQuantizeOnWeightsWithUnsupportedChild::getTestCaseName);
