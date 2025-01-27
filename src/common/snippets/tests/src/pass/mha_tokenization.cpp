// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/mha_tokenization.hpp>
#include <subgraph_mha.hpp>
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/extract_reshapes_from_mha.hpp"

namespace ov {
namespace test {
namespace snippets {

void TokenizeMHASnippetsTests::run() {
    ASSERT_TRUE(model);
    manager.register_pass<ov::snippets::pass::ExtractReshapesFromMHA>();
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeMHASnippets>(config);
    manager.register_pass<ov::snippets::pass::CommonOptimizations>(config);
    disable_rt_info_check();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Dynamic) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}), true, false);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Partially_Dynamic) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 1, -1, 1}, {1, 128, 12, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}), true, false);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Partially_Dynamic_Disabled_Dynamic) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 1, -1, 1}, {1, 128, 12, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}), true, false);
    model = f.getOriginal();
    config = ov::snippets::pass::SnippetsTokenization::Config{1, std::numeric_limits<size_t>::max(), true, true, false, { 3, 4 }};
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Dynamic_M) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, -1, 12, 64}, {1, 128, 12, 64}, {1, 12, -1, 128}, {1, 128, 12, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}), true, false);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_Dynamic_K) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, 128, 12, -1}, {1, 128, 12, -1}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}), true, false);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_3D) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{128, 12, 64}, {128, 12, 64}, {12, 128, 128}, {128, 12, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_3D_Dynamic) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}), true, false);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_MatMul0_Transpose) {
    const auto &f = MHAMatMul0TransposeFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_MatMul0_Transpose_Dynamic) {
    const auto &f = MHAMatMul0TransposeFunction(std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
                                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}),
                                                false);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_with_int_Matmuls) {
    const auto &f = MHAINT8MatMulTypeRelaxedFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}});
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_extraction) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 128, 12, 64}}, true);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Dynamic_Transpose_extraction) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}}, true);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_extraction_and_unsupported_existing_transpose) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 12, 64, 128}, {1, 128, 12, 64}}, true,
                                               std::vector<int64_t>{0, 3, 1, 2});
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Dynamic_Transpose_extraction_and_unsupported_existing_transpose) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}}, true,
                                               std::vector<int64_t>{0, 3, 1, 2});
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Transpose_fusion) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 64, 128, 12}, {1, 128, 12, 64}}, false,
                                               std::vector<int64_t>{0, 2, 1, 3});
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Dynamic_Transpose_fusion) {
    const auto& f = MHATransposedInputFunction(std::vector<PartialShape>{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}}, false,
                                               std::vector<int64_t>{0, 2, 1, 3});
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_4D_SplitM_DynamicParameter) {
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 16, 128, -1}, {1, 128, 16, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}), false, false);
    model = f.getOriginal();
    model_ref = f.getReference();
    config.set_concurrency(32);
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Reshape_extraction) {
    const auto& f = MHAWithExtractedReshapeFunction(std::vector<PartialShape>{{400, 196, 80},
                                                                              {400, 80, 196},
                                                                              {400, 14, 14, 14},
                                                                              {400, 14, 14, 1, 14},
                                                                              {400, 196, 80}}, true);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
