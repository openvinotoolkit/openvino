// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/softmax.hpp"
#include "subgraph_softmax.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string SoftmaxBase::getTestCaseName(const testing::TestParamInfo<ov::test::snippets::SoftmaxParams>& obj) {
    const auto& [inputShapes, axis, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        result << "IS[" << i << "]=" << inputShapes[i] << "_";
    }
    result << "Axis=" << axis << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void SoftmaxBase::SetUp() {
    const auto& [inputShapes, axis, ref_num_nodes_tmp, ref_num_subgraphs_tmp, targetDevice_tmp] = this->GetParam();
    ref_num_nodes = ref_num_nodes_tmp;
    ref_num_subgraphs = ref_num_subgraphs_tmp;
    targetDevice = targetDevice_tmp;
    init_input_shapes(inputShapes);

    auto f = get_subgraph(inputDynamicShapes, axis);
    function = f->getOriginal();

    setIgnoreCallbackMode();
}
std::shared_ptr<SnippetsFunctionBase> Softmax::get_subgraph(const std::vector<PartialShape>& inputShapes, int axis) const {
    return std::make_shared<SoftmaxFunction>(inputShapes, axis);
}

std::shared_ptr<SnippetsFunctionBase> AddSoftmax::get_subgraph(const std::vector<PartialShape>& inputShapes, int axis) const {
    return std::make_shared<AddSoftmaxFunction>(inputShapes, axis);
}

std::shared_ptr<SnippetsFunctionBase> SoftmaxSum::get_subgraph(const std::vector<PartialShape>& inputShapes, int axis) const {
    return std::make_shared<SoftmaxSumFunction>(inputShapes, axis);
}

TEST_P(Softmax, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(AddSoftmax, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(SoftmaxSum, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
