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
    std::ostringstream result;
    result << "IS=" << std::get<0>(obj.param);
    return result.str();
}

void OnlineSoftmaxDecompositionTest::SetUp() {
    LoweringTests::SetUp();
    std::vector<PartialShape> input_shapes{std::get<0>(this->GetParam())};

    snippets_model = std::make_shared<OnlineSoftmaxFunction>(input_shapes);
    manager.register_pass<ov::snippets::pass::OnlineSoftmaxDecomposition>();
}

TEST_P(OnlineSoftmaxDecompositionTest, OnlineSoftmaxDecomposition) {
    model = snippets_model->getOriginal();
    model_ref = snippets_model->getReference();
}

namespace OnlineSoftmaxDecompositionTestInstantiation {
const std::vector<ov::PartialShape> input_shapes{{2, 2, 64, 4096}, {1, 3, 128}, {4, 64}, {32}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_OnlineSoftmaxDecomposition,
                         OnlineSoftmaxDecompositionTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes)),
                         OnlineSoftmaxDecompositionTest::getTestCaseName);

}  // namespace OnlineSoftmaxDecompositionTestInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
