// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/gn_tokenization.hpp>
#include "snippets/pass/gn_tokenization.hpp"
#include "common_test_utils/common_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string TokenizeGNSnippetsTests::getTestCaseName(testing::TestParamInfo<GroupNormalizationParams> obj) {
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

void TokenizeGNSnippetsTests::SetUp() {
    TransformationTestsF::SetUp();
    PartialShape data_shape;
    size_t num_group;
    float eps;
    std::tie(data_shape, num_group, eps) = this->GetParam();
    OPENVINO_ASSERT(data_shape.size() >= 2, "First input rank for group normalization op should be greater than 1");
    PartialShape scaleShiftShape = PartialShape{data_shape[1]};
    std::vector<PartialShape> input_shapes = { data_shape, scaleShiftShape, scaleShiftShape};
    snippets_model = std::make_shared<GroupNormalizationFunction>(input_shapes, num_group, eps);
    manager.register_pass<ov::snippets::pass::TokenizeGNSnippets>();
}

TEST_P(TokenizeGNSnippetsTests, smoke_TokenizeGNSnippets) {
    model = snippets_model->getOriginal();
    model_ref = snippets_model->getReference();
}

namespace TokenizeGNSnippetsTestsInstantiation {

static const std::vector<ov::PartialShape> input_shapes{{3, 10},
                                                        {3, 10, 1},
                                                        {3, 10, 2, 2},
                                                        {1, 20, 2, 2, 3},
                                                        {1, 20, 2, 2, 3, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GNTokenize,
                         TokenizeGNSnippetsTests,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(5),
                                            ::testing::Values(0.0001)),
                         TokenizeGNSnippetsTests::getTestCaseName);

}  // namespace TokenizeGNSnippetsTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
