// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "pass/gated_mlp_tokenization.hpp"

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/gated_mlp_tokenization.hpp"
#include "snippets/pass/tokenization.hpp"
#include "subgraph_gated_mlp.hpp"

namespace ov {
namespace test {
namespace snippets {

void TokenizeGatedMLPSnippetsTests::run() {
    ASSERT_TRUE(model);
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeGatedMLPSnippets>(base_config);
    manager.register_pass<ov::snippets::pass::CommonOptimizations>(common_config);
    disable_rt_info_check();

    // To avoid Transpose extraction from MatMuls
    manager.get_pass_config()->set_callback<ov::snippets::pass::ExplicitTransposeMatMulInputs>(
        [](const std::shared_ptr<const ov::Node>& n) {
            return true;
        });
}

using TokenizeGatedMLPSnippetsParam = std::tuple<
    PartialShape,
    std::vector<Shape>,
    GatedMLPFunction::WeightFormat,
    ov::test::utils::ActivationTypes
>;
class TokenizeGatedMLPSnippetsParamTests : public TokenizeGatedMLPSnippetsTests,
                                           public testing::WithParamInterface<TokenizeGatedMLPSnippetsParam> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        auto [inputShape, weightsShapes, weightFormat, ActType] = GetParam();

        const auto& f = GatedMLPFunction({inputShape}, weightsShapes, weightFormat, ActType);
        model = f.getOriginal();

        // Currently we support only Constants on second inputs of MatMuls
        // Weights decompression in not supported and not tokenized
        if (weightFormat == GatedMLPFunction::WeightFormat::FP32) {
            model_ref = f.getReference();
        }
    }
};

TEST_P(TokenizeGatedMLPSnippetsParamTests, GatedMLP) {
    run();
}

static std::string getTestCaseName(const testing::TestParamInfo<TokenizeGatedMLPSnippetsParam>& info) {
    auto [inputShape, weightsShapes, weightFormat, ActType] = info.param;
    std::ostringstream result;
    result << "InputShape=" << ov::test::utils::partialShape2str({inputShape}) << "_";
    result << "weightsShapes=" << ov::test::utils::vec2str(weightsShapes) << "_";
    result << "WeightFormat=" << weightFormat << "_";
    result << "ActType=" << ActType;
    return result.str();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_GatedMLP,
    TokenizeGatedMLPSnippetsParamTests,
    testing::Combine(
        testing::Values(PartialShape{-1, -1, 896}),
        testing::Values(std::vector<Shape>{{4864, 896}, {4864, 896}, {896, 4864}}),
        testing::Values(GatedMLPFunction::WeightFormat::FP32, GatedMLPFunction::WeightFormat::FP16),
        testing::Values(utils::ActivationTypes::Swish, utils::ActivationTypes::Relu)),
    getTestCaseName);

}  // namespace snippets
}  // namespace test
}  // namespace ov
