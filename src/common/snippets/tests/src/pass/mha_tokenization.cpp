// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/mha_tokenization.hpp>
#include <subgraph_mha.hpp>
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/matmul_transpose.hpp"

namespace ov {
namespace test {
namespace snippets {

void TokenizeMHASnippetsTests::run() {
    ASSERT_TRUE(function);
    std::string name;
    manager.register_pass<ngraph::snippets::pass::EnumerateNodes>();
    manager.register_pass<ngraph::snippets::pass::TokenizeMHASnippets>();
    manager.register_pass<ngraph::snippets::pass::MatMulTranspose>();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_MatMul0_Transpose) {
    const auto &f = MHAMatMul0TransposeFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov