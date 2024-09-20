// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/softmax_decomposition.hpp"
#include "common_test_utils/common_utils.hpp"
#include "subgraph_softmax.hpp"
#include "subgraph_lowered.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string SoftmaxDecompositionTest::getTestCaseName(testing::TestParamInfo<SoftmaxDecompositionTestParams> obj) {
    PartialShape input_shape;
    int axis;
    SoftmaxVersion softmax_version;
    std::tie(input_shape, axis, softmax_version) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({input_shape}) << "_";
    result << "axis=" << axis << "_";
    result << "softmax_version=" << softmax_version;
    return result.str();
}

void SoftmaxDecompositionTest::SetUp() {
    LoweringTests::SetUp();
    std::vector<PartialShape> input_shapes{{}};
    int axis;
    SoftmaxVersion softmax_version;
    std::tie(input_shapes[0], axis, softmax_version) = this->GetParam();
    snippets_model = std::make_shared<SoftmaxFunction>(input_shapes, axis, softmax_version);
}

TEST_P(SoftmaxDecompositionTest, SoftmaxDecomposition) {
    auto subgraph = getLoweredSubgraph(snippets_model->getOriginal());
    model = subgraph->body_ptr();
    model_ref = snippets_model->getLowered();
}

namespace SoftmaxDecompositionTestInstantiation {
const std::vector<ov::PartialShape> input_shapes{{1, 3, 256, 256}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_SoftmaxDecomposition_positive_axis,
                         SoftmaxDecompositionTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(3),
                                            ::testing::Values(SoftmaxVersion::V1, SoftmaxVersion::V8)),
                         SoftmaxDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_SoftmaxDecomposition_negative_axis,
                         SoftmaxDecompositionTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(-1),
                                            ::testing::Values(SoftmaxVersion::V8)),
                         SoftmaxDecompositionTest::getTestCaseName);

}  // namespace SoftmaxDecompositionTestInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
