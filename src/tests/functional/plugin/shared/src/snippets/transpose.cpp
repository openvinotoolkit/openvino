// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/transpose.hpp"
#include "subgraph_transpose.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Transpose::getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeParams> obj) {
    ov::PartialShape inputShape;
    std::vector<int> order;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShape, order, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({inputShape}) << "_";
    result << "Order=" << ov::test::utils::vec2str(order) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Transpose::SetUp() {
    ov::PartialShape inputShape;
    std::vector<int> order;
    std::tie(inputShape, order, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{inputShape}, {inputShape.get_shape(), }}});

    auto f = ov::test::snippets::TransposeFunction({inputShape}, order);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

std::string TransposeMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMulParams> obj) {
    std::vector<ov::PartialShape> inputShapes(2);
    std::vector<int> order;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes[0], inputShapes[1], order, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (int i = 0; i < inputShapes.size(); i++)
        result << "IS[" << i << "]=" << ov::test::utils::partialShape2str({inputShapes[i]}) << "_";
    result << "Order=" << ov::test::utils::vec2str(order) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void TransposeMul::SetUp() {
    std::vector<ov::PartialShape> inputShapes(2);
    std::vector<int> order;
    std::tie(inputShapes[0], inputShapes[1], order, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));
    auto f = ov::test::snippets::TransposeMulFunction(inputShapes, order);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

TEST_P(Transpose, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(TransposeMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
