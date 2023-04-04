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
    result << "IS=" << CommonTestUtils::partialShape2str({inputShape}) << "_";
    result << "Order=" << CommonTestUtils::vec2str(order) << "_";
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

TEST_P(Transpose, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
