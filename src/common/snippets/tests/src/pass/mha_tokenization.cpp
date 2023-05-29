// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/mha_tokenization.hpp>
#include <subgraph_mha.hpp>
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/common_optimizations.hpp"

namespace ov {
namespace test {
namespace snippets {

void TokenizeMHASnippetsTests::run() {
    ASSERT_TRUE(function);
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeMHASnippets>();
    manager.register_pass<ov::snippets::pass::CommonOptimizations>();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_MatMul0_Transpose) {
    const auto &f = MHAMatMul0TransposeFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_int_Matmuls) {
    const auto &f = MHAINT8MatMulTypeRelaxedFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_extraction) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 128, 12, 64}}, true);
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_extraction_and_unsupported_existing_transpose) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 12, 64, 128}, {1, 128, 12, 64}}, true,
                                               std::vector<int64_t>{0, 3, 1, 2});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_fusion) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 64, 128, 12}, {1, 128, 12, 64}}, false,
                                               std::vector<int64_t>{0, 2, 1, 3});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}



}  // namespace snippets
}  // namespace test
}  // namespace ov