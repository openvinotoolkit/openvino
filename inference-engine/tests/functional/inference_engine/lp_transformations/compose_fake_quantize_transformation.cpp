// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include <transformations/utils/utils.hpp>
#include <low_precision/network_helper.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/compose_fake_quantize_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class ComposeFakeQuantizeTransformationParams {
public:
    class Values {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
    };

    ngraph::element::Type originalPrecision;
    Values actual;
    Values expected;
};

typedef std::tuple<
    ngraph::Shape,
    ComposeFakeQuantizeTransformationParams> ComposeFakeQuantizeTransformationValues;

class ComposeFakeQuantizeTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<ComposeFakeQuantizeTransformationValues> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());
        actualFunction = ngraph::builder::subgraph::ComposeFakeQuantizeFunction::get(
            testValues.originalPrecision,
            inputShape,
            testValues.actual.fakeQuantize,
            testValues.actual.dequantization1,
            testValues.actual.dequantization2);

        const auto input = actualFunction->get_parameters()[0];
        const auto fakeQuantizes = input->output(0).get_target_inputs();
        const auto it = fakeQuantizes.begin();
        const auto fakeQuantize = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(it->get_node()->shared_from_this());
        low_precision::NetworkHelper::composeFakeQuantize(fakeQuantize);

        referenceFunction = ngraph::builder::subgraph::ComposeFakeQuantizeFunction::get(
            testValues.originalPrecision,
            inputShape,
            testValues.expected.fakeQuantize,
            testValues.expected.dequantization1,
            testValues.expected.dequantization2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ComposeFakeQuantizeTransformationValues> obj) {
        const auto inputShape = std::get<0>(obj.param);
        const auto testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            testValues.originalPrecision << "_" <<
            inputShape << "_" <<
            testValues.actual.fakeQuantize << "_" <<
            testValues.actual.dequantization1 << "_" <<
            testValues.actual.dequantization2 << "_" <<
            testValues.expected.fakeQuantize << "_" <<
            testValues.expected.dequantization1 << "_" <<
            testValues.expected.dequantization2;
        return result.str();
    }
};

TEST_P(ComposeFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> inputShapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

const std::vector<ComposeFakeQuantizeTransformationParams> testValues = {
    {
        ngraph::element::f32,
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
            { {ngraph::element::f32},  {}, { 0.01f } },
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {},
            {}
        },
    },
    {
        ngraph::element::f32,
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {ngraph::element::f32},  {-128}, { 0.01f } },
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {},
            {}
        },
    },
    {
        ngraph::element::f32,
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {ngraph::element::f32},  {-128}, { 0.01f } },
            { {ngraph::element::f32},  {-128}, { 0.01f } }
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {ngraph::element::f32},  {-128}, { 0.01f } },
            { {ngraph::element::f32},  {-128}, { 0.01f } }
        },
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ComposeFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    ComposeFakeQuantizeTransformation::getTestCaseName);
