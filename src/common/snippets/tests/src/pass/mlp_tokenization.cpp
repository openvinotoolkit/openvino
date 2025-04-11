// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <pass/mlp_tokenization.hpp>
#include <subgraph_mlp.hpp>

#include "openvino/pass/serialize.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/extract_reshapes_from_mha.hpp"
#include "snippets/pass/mlp_seq_tokenization.hpp"
#include "snippets/pass/tokenization.hpp"
namespace ov {
namespace test {
namespace snippets {

void TokenizeMLPSnippetsTests::run() {
    ASSERT_TRUE(model);
    manager.register_pass<ov::snippets::pass::ExtractReshapesFromMHA>();
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeMLPSeqSnippets>(config);
    manager.register_pass<ov::snippets::pass::CommonOptimizations>(config);
    disable_rt_info_check();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_2D_f32) {
    const auto& f =
        MLPSeqFunction(std::vector<PartialShape>{{64, 64}}, std::vector<ov::element::Type>({ov::element::f32}), 1);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_2D_i8) {
    const auto& f =
        MLPSeqFunction(std::vector<PartialShape>{{64, 64}}, std::vector<ov::element::Type>({ov::element::f32}), 10);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_2D_f32_rect_matrix) {
    const auto& f =
        MLPSeqFunction(std::vector<PartialShape>{{128, 128}}, std::vector<ov::element::Type>({ov::element::f32}), 10);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_2D_f32_rect_matrix_2) {
    const auto& f =
        MLPSeqFunction(std::vector<PartialShape>{{128, 128}}, std::vector<ov::element::Type>({ov::element::f32}), 10);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_4D_f32_rect_matrix) {
    const auto& f = MLPSeqFunction(std::vector<PartialShape>{{1, 4, 128, 128}},
                                   std::vector<ov::element::Type>({ov::element::f32}),
                                   10);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_4D_f32_rect_matrix_2) {
    const auto& f = MLPSeqFunction(std::vector<PartialShape>{{1, 2, 128, 128}},
                                   std::vector<ov::element::Type>({ov::element::f32}),
                                   10);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
