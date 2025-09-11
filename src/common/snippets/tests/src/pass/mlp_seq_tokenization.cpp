// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <pass/mlp_seq_tokenization.hpp>
#include <subgraph_mlp_seq.hpp>

#include "common_test_utils/common_utils.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/mlp_seq_tokenization.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace test {
namespace snippets {

void TokenizeMLPSeqSnippetsTests::run() {
    ASSERT_TRUE(model);
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeMLPSeqSnippets>(mlp_seq_config);
    manager.register_pass<ov::snippets::pass::CommonOptimizations>(common_config);
    disable_rt_info_check();
}
class TokenizeMLPSeqSnippetsParamTests : public TokenizeMLPSeqSnippetsTests,
                                         public testing::WithParamInterface<std::tuple<std::vector<PartialShape>, ov::element::Type, int>> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        auto params = GetParam();
        auto shape = std::get<0>(params);
        auto elem_type = std::get<1>(params);
        auto hidden_layers = std::get<2>(params);

        const auto& f = MLPSeqQuantizedTypeRelaxedFunction(shape,
                                                          std::vector<ov::element::Type>({elem_type}),
                                                          hidden_layers,
                                                          128);
        model = f.getOriginal();
        model_ref = f.getReference();
    }
};

TEST_P(TokenizeMLPSeqSnippetsParamTests, TypeRelaxed_2D) {
    run();
}

static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<std::vector<PartialShape>, ov::element::Type, int>>& info) {
    std::vector<PartialShape> shape = std::get<0>(info.param);
    ov::element::Type elem_type = std::get<1>(info.param);
    int hidden_layers = std::get<2>(info.param);
    std::ostringstream result;
    result << "InputShape=" << ov::test::utils::partialShape2str(shape) << "_";
    result << "ElementType=" << elem_type.get_type_name() << "_";
    result << "HiddenLayers=" << hidden_layers << "_";
    result << "Precision=" << elem_type.get_type_name() << "_";
    return result.str();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MLP_SEQ,
    TokenizeMLPSeqSnippetsParamTests,
    testing::Combine(
        testing::Values(std::vector<PartialShape>{{64, 64}}, std::vector<PartialShape>{{128, 128}}),
        testing::Values(ov::element::f32, ov::element::u8),
        testing::Values(1, 2, 3, 5, 7)),
    getTestCaseName);

}  // namespace snippets
}  // namespace test
}  // namespace ov
