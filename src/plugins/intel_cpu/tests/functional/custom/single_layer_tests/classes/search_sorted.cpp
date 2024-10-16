// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "search_sorted.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/precision_support.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace SearchSorted {

static const int SEED = 7877;

std::string SearchSortedLayerCPUTest::getTestCaseName(testing::TestParamInfo<SearchSortedLayerTestParams> obj) {
    SearchSortedLayerTestParams basicParamsSet;
    basicParamsSet = obj.param;

    SearchSortedSpecificParams searchSortedParams;

    ElementType inputPrecision;
    std::tie(searchSortedParams, inputPrecision) = basicParamsSet;

    InputShape sortedInputShape;
    InputShape valuesInputShape;
    bool right_mode;
    std::tie(sortedInputShape, valuesInputShape, right_mode) = searchSortedParams;

    std::ostringstream result;
    result << inputPrecision << "_IS=";
    result << ov::test::utils::partialShape2str({sortedInputShape.first}) << ",";
    result << ov::test::utils::partialShape2str({valuesInputShape.first}) << "_";
    result << "TS=";
    result << "(";
    for (const auto& targetShape : sortedInputShape.second) {
        result << ov::test::utils::vec2str(targetShape) << "_";
    }
    result << ", ";
    for (const auto& targetShape : valuesInputShape.second) {
        result << ov::test::utils::vec2str(targetShape) << "_";
    }
    result << ")_";
    result << "right_mode=" << right_mode;

    return result.str();
}

void SearchSortedLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();

    const auto dataPrecision = funcInputs[0].get_element_type();

    auto sortedTensor =
        ov::test::utils::create_and_fill_tensor_unique_sequence(dataPrecision, targetInputStaticShapes[0], 0, 8, SEED);
    inputs.insert({funcInputs[0].get_node_shared_ptr(), sortedTensor});

    auto valuesTensor =
        ov::test::utils::create_and_fill_tensor_unique_sequence(dataPrecision, targetInputStaticShapes[1], 0, 8, SEED);

    inputs.insert({funcInputs[1].get_node_shared_ptr(), valuesTensor});
}

void SearchSortedLayerCPUTest::SetUp() {
    SearchSortedLayerTestParams basicParamsSet;
    basicParamsSet = this->GetParam();

    SearchSortedSpecificParams searchSortedParams;

    ElementType inputPrecision;
    std::tie(searchSortedParams, inputPrecision) = basicParamsSet;

    selectedType = makeSelectedTypeStr("ref", inputPrecision);
    targetDevice = ov::test::utils::DEVICE_CPU;

    InputShape sortedInputShape;
    InputShape valuesInputShape;
    bool right_mode;
    std::tie(sortedInputShape, valuesInputShape, right_mode) = searchSortedParams;

    init_input_shapes({sortedInputShape, valuesInputShape});
    auto sortedParam = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[0]);
    auto valuesParam = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[1]);

    auto op = std::make_shared<ov::op::v15::SearchSorted>(sortedParam, valuesParam, right_mode);

    ov::ParameterVector params{sortedParam, valuesParam};
    function = makeNgraphFunction(inputPrecision, params, op, "SearchSorted");
}

TEST_P(SearchSortedLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "SearchSorted");
}

const std::vector<SearchSortedSpecificParams> SearchSortedParamsVector = {
    SearchSortedSpecificParams{InputShape{{}, {{1, 18, 104}}}, InputShape{{}, {{1, 18, 104}}}, true},
};

}  // namespace SearchSorted
}  // namespace test
}  // namespace ov
