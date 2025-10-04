// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include <gmock/gmock.h>

#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

using ov::snippets::lowered::Expression;
using ov::snippets::lowered::ExpressionPtr;
using ov::snippets::lowered::LoopManager;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

ExpressionPtr make_expression_with_loops(const std::vector<size_t>& loop_ids) {
    auto expr = std::make_shared<Expression>();
    expr->set_loop_ids(loop_ids);
    return expr;
}

TEST(LoopManagerTests, SingleExpressionKeepsAllLoops) {
    auto expr = make_expression_with_loops({1, 2, 3});

    LoopManager manager;
    const auto loops = manager.get_common_outer_loops({expr});

    EXPECT_THAT(loops, ElementsAre(1, 2, 3));
}

TEST(LoopManagerTests, MultipleExpressionsShrinkToCommonPrefix) {
    auto expr0 = make_expression_with_loops({0, 1, 2});
    auto expr1 = make_expression_with_loops({0, 1, 3});
    auto expr2 = make_expression_with_loops({0, 4});

    LoopManager manager;
    const auto loops = manager.get_common_outer_loops({expr0, expr1, expr2});

    EXPECT_THAT(loops, ElementsAre(0));
}

TEST(LoopManagerTests, ExpressionsWithoutCommonLoops) {
    auto expr0 = make_expression_with_loops({5, 6});
    auto expr1 = make_expression_with_loops({7, 8});

    LoopManager manager;
    const auto loops = manager.get_common_outer_loops({expr0, expr1});

    EXPECT_THAT(loops, IsEmpty());
}

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
