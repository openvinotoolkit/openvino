// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/eltwise_two_results.hpp"
#include "subgraph_simple.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string EltwiseTwoResults::getTestCaseName(testing::TestParamInfo<ov::test::snippets::EltwiseTwoResultsParams> obj) {
    InputShape inputShapes0, inputShapes1;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes0, inputShapes1, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({inputShapes0.first}) << "_"
           << ov::test::utils::partialShape2str({inputShapes1.first}) << "_";

    result << "TS[0]=";
    for (const auto& item : inputShapes0.second) {
        result << ov::test::utils::vec2str(item) << "_";
    }
    result << "TS[1]=";
    for (const auto& item : inputShapes1.second) {
        result << ov::test::utils::vec2str(item) << "_";
    }
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EltwiseTwoResults::SetUp() {
    InputShape inputShape0, inputShape1;
    std::tie(inputShape0, inputShape1, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({inputShape0, inputShape1});

    auto f = ov::test::snippets::EltwiseTwoResultsFunction({inputDynamicShapes[0], inputDynamicShapes[1]});
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

TEST_P(EltwiseTwoResults, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
