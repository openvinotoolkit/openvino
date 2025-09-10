// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/online_softmax_decomposition.hpp"
#include "snippets/pass/online_softmax_decomposition.hpp"
#include "common_test_utils/common_utils.hpp"
#include "subgraph_softmax.hpp"
#include "subgraph_lowered.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string OnlineSoftmaxDecompositionTest::getTestCaseName(testing::TestParamInfo<OnlineSoftmaxDecompositionTestParams> obj) {
    const auto& [input_shape] = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({input_shape}) << "_";
    return result.str();
}

void OnlineSoftmaxDecompositionTest::SetUp() {
    TransformationTestsF::SetUp();

    std::vector<PartialShape> input_shapes{{}};
    const auto& [_tmp] = this->GetParam();
    input_shapes[0] = _tmp;

    snippets_model = std::make_shared<OnlineSoftmaxFunction>(input_shapes);
    manager.register_pass<ov::snippets::pass::OnlineSoftmaxDecomposition>();
}

TEST_P(OnlineSoftmaxDecompositionTest, OnlineSoftmaxDecomposition) {
    model = snippets_model->getOriginal();
    model_ref = snippets_model->getReference();
}

namespace OnlineSoftmaxDecompositionTestInstantiation {
const std::vector<ov::PartialShape> input_shapes{{2, 2, 64, 4096}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_OnlineSoftmaxDecomposition,
                         OnlineSoftmaxDecompositionTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes)),
                         OnlineSoftmaxDecompositionTest::getTestCaseName);

}  // namespace OnlineSoftmaxDecompositionTestInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
