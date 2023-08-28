// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/max_num_params_eltwise.hpp"
#include "subgraph_simple.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string MaxNumParamsEltwise::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MaxNumParamsEltwiseParams> obj) {
    ov::test::InputShape inputShapes;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
    result << "TS[0]=";
    for (const auto& item : inputShapes.second) {
        result << ov::test::utils::vec2str(item) << "_";
    }

    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MaxNumParamsEltwise::SetUp() {
    ov::test::InputShape inputShape;
    std::tie(inputShape, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    std::vector<ov::test::InputShape> expandedShapes(10, inputShape);
    init_input_shapes(expandedShapes);

    auto f = ov::test::snippets::EltwiseMaxNumParamsFunction(inputDynamicShapes);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

TEST_P(MaxNumParamsEltwise, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
