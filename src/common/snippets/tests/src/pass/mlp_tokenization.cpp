// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <pass/mlp_tokenization.hpp>
#include <subgraph_mlp.hpp>

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

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_SEQ_TypeRelaxed_2D_f32_HL2) {
    const auto& f = MLPSeqQuantizedTypeRelaxedFunction(std::vector<PartialShape>{{64, 64}},
                                                       std::vector<ov::element::Type>({ov::element::f32}),
                                                       2,
                                                       2);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_SEQ_TypeRelaxed_2D_f32_HL3) {
    const auto& f = MLPSeqQuantizedTypeRelaxedFunction(std::vector<PartialShape>{{64, 64}},
                                                       std::vector<ov::element::Type>({ov::element::f32}),
                                                       2,
                                                       3);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_SEQ_TypeRelaxed_2D_f32_HL5) {
    const auto& f = MLPSeqQuantizedTypeRelaxedFunction(std::vector<PartialShape>{{64, 64}},
                                                       std::vector<ov::element::Type>({ov::element::f32}),
                                                       2,
                                                       5);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_SEQ_TypeRelaxed_2D_f32_HL7) {
    const auto& f = MLPSeqQuantizedTypeRelaxedFunction(std::vector<PartialShape>{{64, 64}},
                                                       std::vector<ov::element::Type>({ov::element::f32}),
                                                       2,
                                                       7);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_SEQ_TypeRelaxed_2D_i8_HL2) {
    const auto& f = MLPSeqQuantizedTypeRelaxedFunction(std::vector<PartialShape>{{64, 64}},
                                                       std::vector<ov::element::Type>({ov::element::u8}),
                                                       2,
                                                       2);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_SEQ_TypeRelaxed_2D_i8_HL3) {
    const auto& f = MLPSeqQuantizedTypeRelaxedFunction(std::vector<PartialShape>{{64, 64}},
                                                       std::vector<ov::element::Type>({ov::element::u8}),
                                                       2,
                                                       3);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_SEQ_TypeRelaxed_2D_i8_HL5) {
    const auto& f = MLPSeqQuantizedTypeRelaxedFunction(std::vector<PartialShape>{{64, 64}},
                                                       std::vector<ov::element::Type>({ov::element::u8}),
                                                       2,
                                                       5);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

TEST_F(TokenizeMLPSnippetsTests, smoke_Snippets_MLP_SEQ_TypeRelaxed_2D_i8_HL7) {
    const auto& f = MLPSeqQuantizedTypeRelaxedFunction(std::vector<PartialShape>{{64, 64}},
                                                       std::vector<ov::element::Type>({ov::element::u8}),
                                                       2,
                                                       7);
    model = f.getOriginal();
    model_ref = f.getReference();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
