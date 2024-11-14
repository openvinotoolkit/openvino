// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/gn_decomposition.hpp"
#include "common_test_utils/common_utils.hpp"
#include "subgraph_group_normalization.hpp"
#include "subgraph_lowered.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string GNDecompositionTest::getTestCaseName(testing::TestParamInfo<GroupNormalizationParams> obj) {
    PartialShape input_shape;
    size_t num_group;
    float eps;
    std::tie(input_shape, num_group, eps) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({input_shape}) << "_";
    result << "num_group=" << num_group << "_";
    result << "eps=" << eps;
    return result.str();
}

void GNDecompositionTest::SetUp() {
    LoweringTests::SetUp();
    PartialShape data_shape;
    size_t num_group;
    float eps;
    std::tie(data_shape, num_group, eps) = this->GetParam();
    OPENVINO_ASSERT(data_shape.size() >= 2, "First input rank for group normalization op should be greater than 1");
    PartialShape scaleShiftShape = PartialShape{data_shape[1]};
    std::vector<PartialShape> input_shapes = { data_shape, scaleShiftShape, scaleShiftShape};
    snippets_model = std::make_shared<GroupNormalizationFunction>(input_shapes, num_group, eps);
}

TEST_P(GNDecompositionTest, GNDecomposition) {
    auto subgraph = getLoweredSubgraph(snippets_model->getOriginal());
    model = subgraph->body_ptr();
    model_ref = snippets_model->getLowered();
}

namespace GNDecompositionTestInstantiation {

const std::vector<ov::PartialShape> input_shapes{{1, 8},
                                                 {1, 8, 18},
                                                 {1, 16, 8, 5},
                                                 {3, 8, 2, 2, 3},
                                                 {3, 8, 2, 2, 3, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GNDecomposition,
                         GNDecompositionTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(4),
                                            ::testing::Values(0.0001)),
                         GNDecompositionTest::getTestCaseName);

}  // namespace GNDecompositionTestInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
