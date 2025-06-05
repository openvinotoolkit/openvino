// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "pass/mlp_tokenization.hpp"

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/mlp_tokenization.hpp"
#include "snippets/pass/tokenization.hpp"
#include "subgraph_mlp.hpp"

namespace ov {
namespace test {
namespace snippets {

void TokenizeMLPSnippetsTests::run() {
    ASSERT_TRUE(model);
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeMLPSnippets>(config);
    manager.register_pass<ov::snippets::pass::CommonOptimizations>(config);
    disable_rt_info_check();

    // To avoid Transpose exraction from MatMuls
    manager.get_pass_config()->set_callback<ov::snippets::pass::ExplicitTransposeMatMulInputs>(
        [](const std::shared_ptr<const ov::Node>& n){
            return true;
        });
}

using TokenizeMLPSnippetsParam = std::tuple<
    PartialShape,
    std::vector<Shape>,
    MLPFunction::WeightFormat,
    ov::test::utils::ActivationTypes
>;
class TokenizeMLPSnippetsParamTests : public TokenizeMLPSnippetsTests,
                                      public testing::WithParamInterface<TokenizeMLPSnippetsParam> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        auto [inputShape, weightsShapes, weightFormat, ActType] = GetParam();

        const auto& f = MLPFunction({inputShape}, weightsShapes, weightFormat, ActType);
        model = f.getOriginal();

        // Currently we support only Constants on second inputs of MatMuls
        // Weights decompression in not supported and not tokenized
        if (weightFormat == MLPFunction::WeightFormat::FP32) {
            model_ref = f.getReference();
        } else {
            model_ref = f.getOriginal();
        }
    }
};

TEST_P(TokenizeMLPSnippetsParamTests, MLP_LLM) {
    run();
}

static std::string getTestCaseName(const testing::TestParamInfo<TokenizeMLPSnippetsParam>& info) {
    auto [inputShape, weightsShapes, weightFormat, ActType] = info.param;
    std::ostringstream result;
    result << "InputShape=" << ov::test::utils::partialShape2str({inputShape}) << "_";
    result << "weightsShapes=" << ov::test::utils::vec2str(weightsShapes) << "_";
    result << "WeightFormat=" << weightFormat << "_";
    result << "ActType=" << ActType;
    return result.str();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MLP,
    TokenizeMLPSnippetsParamTests,
    testing::Combine(
        testing::Values(PartialShape{-1, -1, 896}),
        testing::Values(std::vector<Shape>{{4864, 896}, {4864, 896}, {896, 4864}}),
        testing::Values(MLPFunction::WeightFormat::FP32, MLPFunction::WeightFormat::FP16),
        testing::Values(utils::ActivationTypes::Swish, utils::ActivationTypes::Relu)),
    getTestCaseName);

}  // namespace snippets
}  // namespace test
}  // namespace ov
