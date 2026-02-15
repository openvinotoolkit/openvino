// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "snippets/pass/propagate_precision.hpp"

namespace ov {
namespace test {
namespace snippets {


class PrecisionPropagationGetPrecisionsTest : public testing::Test {};

TEST_F(PrecisionPropagationGetPrecisionsTest, empty) {
    ASSERT_EQ(std::vector<element::Type>{}, ov::snippets::pass::PropagatePrecision::get_precisions({}, {}));
}

TEST_F(PrecisionPropagationGetPrecisionsTest, selected) {
    ASSERT_EQ(
        std::vector<element::Type>({element::f32, element::f32}),
        ov::snippets::pass::PropagatePrecision::get_precisions(
            { element::f32, element::f32 },
            {
                {element::bf16, element::bf16},
                {element::f32, element::f32},
                {element::i8, element::i8},
            }));
}

TEST_F(PrecisionPropagationGetPrecisionsTest, first) {
    ASSERT_EQ(
        std::vector<element::Type>({ element::bf16, element::bf16 }),
        ov::snippets::pass::PropagatePrecision::get_precisions(
            { element::i32, element::i32 },
            {
                {element::bf16, element::bf16},
                {element::f32, element::f32},
                {element::i8, element::i8},
            }));
}

} // namespace snippets
} // namespace test
} // namespace ov
