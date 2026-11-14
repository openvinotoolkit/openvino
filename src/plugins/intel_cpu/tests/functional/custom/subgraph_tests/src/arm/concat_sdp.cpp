// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom/subgraph_tests/src/classes/concat_sdp.hpp"

namespace ov {
namespace test {
namespace {

const std::vector<std::vector<InputShape>> inputShapes = {
    // greedy search
    {
        {{1, 8, -1, 64}, {{1, 8, 10, 64}, {1, 8, 1, 64}, {1, 8, 1, 64}, {1, 8, 20, 64}, {1, 8, 1, 64}}},
        {{1, 8, -1, 64}, {{1, 8, 0, 64}, {1, 8, 10, 64}, {1, 8, 11, 64}, {1, 8, 12, 64}, {1, 8, 32, 64}}},
    },
    // beam search
    {
        {{-1, 8, -1, 64}, {{4, 8, 10, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}}},
        {{-1, 8, -1, 64}, {{4, 8, 0, 64}, {4, 8, 10, 64}, {4, 8, 11, 64}, {4, 8, 12, 64}, {4, 8, 13, 64}}},
    },
    // big batch
    {
        {{-1, 8, -1, 64}, {{129, 8, 10, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}}},
        {{-1, 8, -1, 64}, {{129, 8, 0, 64}, {129, 8, 10, 64}, {129, 8, 11, 64}, {129, 8, 12, 64}, {129, 8, 13, 64}}},
    },
};

const ov::AnyMap cfg_none{};
const ov::AnyMap cfg_u8_sym{
    {"KEY_CACHE_PRECISION", "u8"},
    {"VALUE_CACHE_PRECISION", "u8"},
};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTest,
                         ConcatSDPTest,
                         ::testing::Combine(::testing::Values(ElementType::f16),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(cfg_none, cfg_u8_sym),
                                            ::testing::Values(true, false),
                                            ::testing::Values<int64_t>(8),
                                            ::testing::Values<int64_t>(8)),
                         ConcatSDPTest::getTestCaseName);

// Shifts q and k apart so that every attention score is far below zero. The f16 softmax must still pick the
// row max from the scores; seeding it with 0 made exp() underflow and the attention output NaN.
class ConcatSDPNegativeScoresTest : public ConcatSDPTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        ConcatSDPTest::generate_inputs(targetInputStaticShapes);
        const auto& params = function->get_parameters();
        shift(inputs.at(params[0]), 2.0F);   // q
        shift(inputs.at(params[1]), -2.0F);  // k
        shift(inputs.at(params[3]), -2.0F);  // past k
    }

private:
    static void shift(ov::Tensor& tensor, float delta) {
        auto* data = tensor.data<ov::float16>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            data[i] = ov::float16(static_cast<float>(data[i]) + delta);
        }
    }
};

TEST_P(ConcatSDPNegativeScoresTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPNegativeScoresTest,
                         ConcatSDPNegativeScoresTest,
                         ::testing::Combine(::testing::Values(ElementType::f16),
                                            ::testing::Values(inputShapes[0]),
                                            ::testing::Values(cfg_none, cfg_u8_sym),
                                            ::testing::Values(false),
                                            ::testing::Values<int64_t>(8),
                                            ::testing::Values<int64_t>(8, 2)),
                         ConcatSDPTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
