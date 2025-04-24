// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include "transformations/utils/utils.hpp"
#include "low_precision/network_helper.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/compose_fake_quantize.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class ComposeFakeQuantizeTransformationParams {
public:
    class Values {
    public:
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
        ov::builder::subgraph::DequantizationOperations dequantization1;
        ov::builder::subgraph::DequantizationOperations dequantization2;
    };

    ov::element::Type originalPrecision;
    Values actual;
    Values expected;
};

typedef std::tuple<
    ov::Shape,
    ComposeFakeQuantizeTransformationParams> ComposeFakeQuantizeTransformationValues;

class ComposeFakeQuantizeTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<ComposeFakeQuantizeTransformationValues> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());
        actualFunction = ov::builder::subgraph::ComposeFakeQuantizeFunction::get(
            testValues.originalPrecision,
            inputShape,
            testValues.actual.fakeQuantize,
            testValues.actual.dequantization1,
            testValues.actual.dequantization2);

        const auto input = actualFunction->get_parameters()[0];
        const auto fakeQuantizes = input->output(0).get_target_inputs();
        const auto it = fakeQuantizes.begin();
        const auto fakeQuantize = ov::as_type_ptr<ov::op::v0::FakeQuantize>(it->get_node()->shared_from_this());
        ov::pass::low_precision::NetworkHelper::composeFakeQuantize(fakeQuantize);

        referenceFunction = ov::builder::subgraph::ComposeFakeQuantizeFunction::get(
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
    auto res = compare_functions(actualFunction, referenceFunction, true, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::Shape> inputShapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

const std::vector<ComposeFakeQuantizeTransformationParams> testValues = {
    {
        ov::element::f32,
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
            { {ov::element::f32},  {}, { 0.01f } },
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {},
            {}
        },
    },
    {
        ov::element::f32,
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {ov::element::f32},  {-128}, { 0.01f } },
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {},
            {}
        },
    },
    {
        ov::element::f32,
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {ov::element::f32},  {-128}, { 0.01f } },
            { {ov::element::f32},  {-128}, { 0.01f } }
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {ov::element::f32},  {-128}, { 0.01f } },
            { {ov::element::f32},  {-128}, { 0.01f } }
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
