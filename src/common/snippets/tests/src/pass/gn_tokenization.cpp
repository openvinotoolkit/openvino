// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/gn_tokenization.hpp>
#include "subgraph_group_normalization.hpp"
#include "snippets/pass/gn_tokenization.hpp"

namespace ov {
namespace test {
namespace snippets {


void TokenizeGNSnippetsTests::run() {
    ASSERT_TRUE(model);
    manager.register_pass<ov::snippets::pass::TokenizeGNSnippets>();
    disable_rt_info_check();
}

TEST_F(TokenizeGNSnippetsTests, smoke_Snippets_GN_Tokenization_2D) {
    const auto &f = GroupNormalizationFunction(std::vector<PartialShape>{{1, 10}, {10}, {10}}, 2, 0.00001f);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeGNSnippetsTests, smoke_Snippets_GN_Tokenization_3D) {
    const auto &f = GroupNormalizationFunction(std::vector<PartialShape>{{1, 10, 1}, {10}, {10}}, 2, 0.00001f);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeGNSnippetsTests, smoke_Snippets_GN_Tokenization_4D) {
    const auto &f = GroupNormalizationFunction(std::vector<PartialShape>{{1, 8, 8, 8}, {8}, {8}}, 2, 0.00001f);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeGNSnippetsTests, smoke_Snippets_GN_Tokenization_5D) {
    const auto &f = GroupNormalizationFunction(std::vector<PartialShape>{{1, 16, 1, 1, 1}, {16}, {16}}, 4, 0.00001f);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeGNSnippetsTests, smoke_Snippets_GN_Tokenization_6D) {
    const auto &f = GroupNormalizationFunction(std::vector<PartialShape>{{1, 16, 1, 1, 1, 2}, {16}, {16}}, 8, 0.00001f);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
