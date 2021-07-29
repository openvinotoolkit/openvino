// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include "layer_transformation.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/reshape_fully_connected_function.hpp"

using namespace testing;
using namespace ngraph::pass;

namespace {

class ReshapeFullyConnectedTransformationTestValues {
public:
    ngraph::Shape inputShape;
    ngraph::element::Type inputPrecision1;
    ngraph::element::Type inputPrecision2;
    ngraph::element::Type inputPrecision3;
    ngraph::Shape outputShape;
    ngraph::element::Type outputPrecision;
};

class ReshapeFullyConnectedTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<ReshapeFullyConnectedTransformationTestValues> {
public:
    void SetUp() override {
        using namespace ngraph::builder::subgraph;
        const ReshapeFullyConnectedTransformationTestValues testValues = GetParam();

        actualFunction = ReshapeFullyConnectedFunction::getOriginal(
            testValues.inputShape,
            testValues.inputPrecision1,
            testValues.inputPrecision2,
            testValues.inputPrecision3,
            testValues.outputShape,
            testValues.outputPrecision);

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::ReshapeFullyConnected>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(actualFunction);

        referenceFunction = ReshapeFullyConnectedFunction::getReference(
            testValues.inputShape,
            testValues.inputPrecision1,
            testValues.inputPrecision2,
            testValues.inputPrecision3,
            testValues.outputShape,
            testValues.outputPrecision);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReshapeFullyConnectedTransformationTestValues> obj) {
        const ReshapeFullyConnectedTransformationTestValues testValues = obj.param;
        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.inputPrecision1 << "_" <<
            testValues.inputPrecision2 << "_" <<
            testValues.outputShape << "_" <<
            testValues.outputPrecision;
        return result.str();
    }
};

TEST_P(ReshapeFullyConnectedTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

std::vector<ReshapeFullyConnectedTransformationTestValues> testValues = {
    {
        { 1, 1, 2048 },
        ngraph::element::u8,
        ngraph::element::i8,
        ngraph::element::f32,
        { 1, 1000 },
        ngraph::element::f32
    },
    {
        { 1, 1, 2048 },
        ngraph::element::f32,
        ngraph::element::f32,
        ngraph::element::f32,
        { 1, 1000 },
        ngraph::element::f32
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReshapeFullyConnectedTransformation,
    ::testing::ValuesIn(testValues),
    ReshapeFullyConnectedTransformation::getTestCaseName);
} // namespace
