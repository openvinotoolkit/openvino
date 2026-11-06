// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/gn_tokenization.hpp"

#include <gtest/gtest.h>

#include <pass/gn_tokenization.hpp>

#include "common_test_utils/common_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/tokenization.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string TokenizeGNSnippetsTests::getTestCaseName(testing::TestParamInfo<GroupNormalizationParams> obj) {
    const auto& [input_shape, num_group, eps] = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({input_shape}) << "_";
    result << "num_group=" << num_group << "_";
    result << "eps=" << eps;
    return result.str();
}

void TokenizeGNSnippetsTests::SetUp() {
    TransformationTestsF::SetUp();

    const auto& [data_shape, num_group, eps] = this->GetParam();
    OPENVINO_ASSERT(data_shape.size() >= 2, "First input rank for group normalization op should be greater than 1");
    PartialShape scaleShiftShape = PartialShape{data_shape[1]};
    std::vector<PartialShape> input_shapes = {data_shape, scaleShiftShape, scaleShiftShape};
    snippets_model = std::make_shared<GroupNormalizationFunction>(input_shapes, num_group, eps);
    manager.register_pass<ov::snippets::pass::TokenizeGNSnippets>();
}

TEST_P(TokenizeGNSnippetsTests, smoke_TokenizeGNSnippets) {
    model = snippets_model->getOriginal();
    model_ref = snippets_model->getReference();
}

// A node fed by a Completed (GroupNormalization) Subgraph and by a regular Subgraph must not merge the Completed one:
// its body is never cloned, so merging it dereferenced a null body (crash compiling the LTX-2.3 latent upsampler on
// CPU).
//   x -> Swish -> GroupNormalization -> Add
//          `-------------------------->'
TEST(TokenizeGNSnippetsTests, smoke_Snippets_GNTokenize_CompletedSubgraphFeedsTokenizedNode) {
    const size_t channels = 32;
    auto x = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, channels, 2, 8, 12});
    auto swish = std::make_shared<ov::op::v4::Swish>(x);
    auto scale = ov::op::v0::Constant::create(element::f32, Shape{channels}, std::vector<float>(channels, 1.0f));
    auto bias = ov::op::v0::Constant::create(element::f32, Shape{channels}, std::vector<float>(channels, 0.0f));
    auto group_norm = std::make_shared<ov::op::v12::GroupNormalization>(swish, scale, bias, 8, 1e-5f);
    auto add = std::make_shared<ov::op::v1::Add>(group_norm, swish);
    auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{x});

    // the same matcher combination as SnippetsTokenization
    ov::pass::Manager manager;
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    auto tokenization = manager.register_pass<ov::pass::GraphRewrite>();
    tokenization->add_matcher<ov::snippets::pass::TokenizeGNSnippets>();
    tokenization->add_matcher<ov::snippets::pass::TokenizeSnippets>(get_default_tokenization_config());
    ASSERT_NO_THROW(manager.run_passes(model));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    // GroupNormalization stays a separate Completed Subgraph that feeds the Subgraph of the Add
    const auto add_subgraph =
        ov::as_type_ptr<ov::snippets::op::Subgraph>(model->get_results()[0]->get_input_node_shared_ptr(0));
    ASSERT_TRUE(add_subgraph);
    EXPECT_NE(ov::snippets::pass::GetSnippetsSubgraphType(add_subgraph),
              ov::snippets::pass::SnippetsSubgraphType::Completed);
    size_t completed_inputs = 0;
    for (const auto& input : add_subgraph->input_values()) {
        const auto producer = ov::as_type_ptr<ov::snippets::op::Subgraph>(input.get_node_shared_ptr());
        if (producer && ov::snippets::pass::GetSnippetsSubgraphType(producer) ==
                            ov::snippets::pass::SnippetsSubgraphType::Completed) {
            ++completed_inputs;
            EXPECT_TRUE(ov::is_type<ov::op::v12::GroupNormalization>(
                producer->body_ptr()->get_results()[0]->get_input_node_shared_ptr(0)));
        }
    }
    EXPECT_EQ(completed_inputs, 1);
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
