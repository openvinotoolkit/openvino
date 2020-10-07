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
#include <transformations/low_precision/layer_transformation.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

#include "ngraph_functions/low_precision_transformations/separate_in_standalone_branch_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph::pass;

class SeparateInStandaloneBranchParams {
public:
    ngraph::builder::subgraph::DequantizationOperations dequantization;
    size_t numberOfOperations;
    size_t indexOfTargetOperation;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    SeparateInStandaloneBranchParams> SeparateInStandaloneBranchTestValues;

class SeparateInStandaloneBranchTestTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<SeparateInStandaloneBranchTestValues> {
public:
    void SetUp() override {
        const auto inputPrecision = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());
        const SeparateInStandaloneBranchParams testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::SeparateInStandaloneBranchFunction::getOriginal(
            inputPrecision,
            inputShape,
            testValues.dequantization,
            testValues.numberOfOperations);

        const auto targetOperation = actualFunction->get_results()[testValues.indexOfTargetOperation]->get_input_node_shared_ptr(0);
        const auto result = low_precision::LayerTransformation::separateInStandaloneBranch(targetOperation);
        actualFunction = std::make_shared<ngraph::Function>(
            std::make_shared<ngraph::opset1::Result>(result),
            ngraph::ParameterVector{ actualFunction->get_parameters()[0] },
            "SeparateInStandaloneBranchFunction");

        referenceFunction = ngraph::builder::subgraph::SeparateInStandaloneBranchFunction::getReference(
            inputPrecision,
            inputShape,
            testValues.dequantization,
            testValues.numberOfOperations,
            testValues.indexOfTargetOperation);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SeparateInStandaloneBranchTestValues> obj) {
        const auto inputPrecision = std::get<0>(obj.param);
        const auto inputShape = std::get<1>(obj.param);
        const SeparateInStandaloneBranchParams testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result
            << inputPrecision << inputShape << testValues.dequantization
            << "_number_of_operations=" << testValues.numberOfOperations
            << "_index_of_target_operation=" << testValues.indexOfTargetOperation;
        return result.str();
    }
};

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::u8,
    ngraph::element::i8,
};

const std::vector<ngraph::Shape> inputShapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

const std::vector<SeparateInStandaloneBranchParams> testValues = {
    {
        {{ ngraph::element::f32 }, { 128.f }, { 0.1f }},
        2ul,
        0ul
    },
    {
        {{ ngraph::element::f32 }, { 128.f }, { 0.1f }},
        2ul,
        1ul
    },
    {
        {{ ngraph::element::f32 }, { 128.f }, { 0.1f }},
        5ul,
        3ul
    },
    {
        {{ ngraph::element::f32 }, { 128.f }, { 0.1f }},
        1ul,
        0ul
    },
};

TEST_P(SeparateInStandaloneBranchTestTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_CASE_P(
    LPT,
    SeparateInStandaloneBranchTestTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    SeparateInStandaloneBranchTestTransformation::getTestCaseName);

} // namespace
