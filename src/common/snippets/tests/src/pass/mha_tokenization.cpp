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

class SKIP_TokenizeMHASnippetsTests : public TokenizeMHASnippetsTests {
public:
    void SetUp() override {
        GTEST_SKIP();
    }
    void TearDown() override{};
};

void TokenizeMHASnippetsTests::run() {
    ASSERT_TRUE(function);
    manager.register_pass<ov::snippets::pass::ExtractReshapesFromMHA>();
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeMHASnippets>();
    manager.register_pass<ov::snippets::pass::CommonOptimizations>(config);
    disable_rt_info_check();
}

TEST_F(SKIP_TokenizeMHASnippetsTests /* CVS-114607 */, smoke_Snippets_MHA) {
    GTEST_SKIP();
    const auto &f = MHAFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(SKIP_TokenizeMHASnippetsTests /* CVS-114607 */, smoke_Snippets_MHA_with_MatMul0_Transpose) {
    GTEST_SKIP();
    const auto &f = MHAMatMul0TransposeFunction(std::vector<PartialShape>{{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                                                std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::f32}));
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(SKIP_TokenizeMHASnippetsTests /* CVS-114607 */, smoke_Snippets_MHA_with_int_Matmuls) {
    GTEST_SKIP();
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

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_SplitM) {
    const auto& f = MHAWOTransposeSplitMFunction(std::vector<PartialShape>{{10, 9216, 128}, {10, 128, 9216}, {10, 9216, 128}},
                                                 std::vector<ov::element::Type>({ov::element::f32, ov::element::f32, ov::element::f32}),
                                                 std::vector<Shape>{{10, 9, 1024, 128}, {10, 1, 128, 9216}, {10, 1, 9216, 128}, {10, 9216, 128}});
    function = f.getOriginal();
    function_ref = f.getReference();
    config.minimal_concurrency = 18;
    run();
}

TEST_F(SKIP_TokenizeMHASnippetsTests /* CVS-114607 */, smoke_Snippets_MHASelect_SplitM) {
    const auto& f = MHASelectSplitMFunction(std::vector<PartialShape>{{8, 512, 18}, {8, 18, 64}, {1, 512, 64}, {1, 1, 64}, {8, 64, 512}},
                                            std::vector<Shape>{{8, 2, 256, 18}, {8, 1, 18, 64}, {1, 2, 256, 64}, {1, 1, 1, 64},
                                                               {8, 1, 64, 512}, {8, 512, 512}});
    function = f.getOriginal();
    function_ref = f.getReference();
    config.minimal_concurrency = 16;
    run();
}

TEST_F(TokenizeMHASnippetsTests, smoke_Snippets_MHA_Reshape_extraction) {
    const auto& f = MHAWithExtractedReshapeFunction(std::vector<PartialShape>{{400, 196, 80},
                                                                              {400, 80, 196},
                                                                              {400, 14, 14, 14},
                                                                              {400, 14, 14, 1, 14},
                                                                              {400, 196, 80}}, true);
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
