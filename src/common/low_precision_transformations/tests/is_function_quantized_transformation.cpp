// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include "low_precision/low_precision.hpp"

#include <gtest/gtest.h>
#include "ov_lpt_models/common/builders.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class IsFunctionQuantizedTransformationValues {
public:
    ov::Shape shape;
    ov::element::Type precision;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize;
    bool constantSubgraphOnParameters;
    bool inputOnParameters;

    bool isQuantized;
};

class IsFunctionQuantizedTransformation : public LayerTransformation, public testing::WithParamInterface<IsFunctionQuantizedTransformationValues> {
public:
    void SetUp() override {
        const auto testValues = GetParam();

        const auto input = std::make_shared<ov::op::v0::Parameter>(testValues.precision, ov::Shape(testValues.shape));
        const auto fakeQuantize = ov::builder::subgraph::makeFakeQuantize(
            input,
            testValues.precision,
            testValues.fakeQuantize,
            testValues.constantSubgraphOnParameters);

        if (testValues.inputOnParameters) {
            replace_node(fakeQuantize->get_input_node_shared_ptr(3), input);
        }

        ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(fakeQuantize) };
        model = std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "IsFunctionQuantizedFunction");
        model->validate_nodes_and_infer_types();
    }

    static std::string getTestCaseName(testing::TestParamInfo<IsFunctionQuantizedTransformationValues> obj) {
        IsFunctionQuantizedTransformationValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.shape << "_" <<
            testValues.precision << "_" <<
            testValues.fakeQuantize <<
            testValues.constantSubgraphOnParameters << "_" <<
            testValues.inputOnParameters << "_" <<
            testValues.isQuantized;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> model;
};

TEST_P(IsFunctionQuantizedTransformation, Run) {
    const bool isQuantized = ov::pass::low_precision::LowPrecision::isFunctionQuantized(model);

    const auto testValues = GetParam();
    ASSERT_EQ(testValues.isQuantized, isQuantized);
}

const std::vector<ov::Shape> shapes = { ov::Shape({ 1, 3, 72, 48 }) };

const std::vector<IsFunctionQuantizedTransformationValues> testValues = {
    {
        ov::Shape{1, 3, 9, 9},
        ov::element::f32,
        { 255ul, {{ 1, 1, 1, 1 }}, { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
        false,
        false,
        true
    },
    {
        ov::Shape{1, 3, 9, 9},
        ov::element::f32,
        { 255ul, {{ 1, 1, 1, 1 }}, { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
        true,
        false,
        false
    },
    {
        ov::Shape{1, 3, 9, 9},
        ov::element::f32,
        { 255ul, {{ 1, 1, 1, 1 }}, { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
        false,
        true,
        false
    },
    {
        ov::Shape{1, 3, 9, 9},
        ov::element::f32,
        { 255ul, {{ 1, 1, 1, 1 }}, { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
        true,
        true,
        false
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    IsFunctionQuantizedTransformation,
    ::testing::ValuesIn(testValues),
    IsFunctionQuantizedTransformation::getTestCaseName);
