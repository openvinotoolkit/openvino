// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/search_sorted.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

static const int SEED = 7877;

std::string SearchSortedLayerTest::getTestCaseName(const testing::TestParamInfo<SearchSortedLayerTestParams>& obj) {
    SearchSortedLayerTestParams basicParamsSet;
    basicParamsSet = obj.param;

    SearchSortedSpecificParams searchSortedParams;

    ElementType inputPrecision;
    std::string targetDevice;
    std::tie(searchSortedParams, inputPrecision, targetDevice) = basicParamsSet;

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

void SearchSortedLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();

    const auto dataPrecision = funcInputs[0].get_element_type();

    auto sortedTensor =
        ov::test::utils::create_and_fill_tensor_unique_sequence(dataPrecision, targetInputStaticShapes[0], 0, 8, SEED);
    inputs.insert({funcInputs[0].get_node_shared_ptr(), sortedTensor});

    auto valuesTensor = ov::test::utils::create_and_fill_tensor(dataPrecision, targetInputStaticShapes[1]);

    inputs.insert({funcInputs[1].get_node_shared_ptr(), valuesTensor});
}

void SearchSortedLayerTest::SetUp() {
    SearchSortedLayerTestParams basicParamsSet;
    basicParamsSet = this->GetParam();

    SearchSortedSpecificParams searchSortedParams;

    ElementType inputPrecision;
    std::tie(searchSortedParams, inputPrecision, targetDevice) = basicParamsSet;

    InputShape sortedInputShape;
    InputShape valuesInputShape;
    bool right_mode;
    std::tie(sortedInputShape, valuesInputShape, right_mode) = searchSortedParams;

    init_input_shapes({sortedInputShape, valuesInputShape});
    auto sortedParam = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[0]);
    auto valuesParam = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[1]);

    auto op = std::make_shared<ov::op::v15::SearchSorted>(sortedParam, valuesParam, right_mode);

    ov::ParameterVector params{sortedParam, valuesParam};
    function = std::make_shared<ov::Model>(op->outputs(), params, "SearchSorted");
}

const std::vector<SearchSortedSpecificParams> SearchSortedLayerTest::GenerateParams() {
    const std::vector<SearchSortedSpecificParams> params = {
        SearchSortedSpecificParams{InputShape{{}, {{1, 18, 104}}}, InputShape{{}, {{1, 18, 104}}}, true},
        SearchSortedSpecificParams{InputShape{{}, {{1, 2, 3, 100}}}, InputShape{{}, {{1, 2, 3, 10}}}, true},
        SearchSortedSpecificParams{InputShape{{}, {{2, 1, 2, 3, 10}}}, InputShape{{}, {{2, 1, 2, 3, 20}}}, false},
        SearchSortedSpecificParams{InputShape{{}, {{1}}}, InputShape{{}, {{2, 1, 2, 3, 20}}}, false},
        SearchSortedSpecificParams{InputShape{{}, {{50}}}, InputShape{{1, -1, 10}, {{1, 18, 10}}}, false},
    };

    return params;
}

}  // namespace test
}  // namespace ov
