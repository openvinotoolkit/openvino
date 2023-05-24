// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "snippets/pass/propagate_precision.hpp"

namespace ov {
namespace test {
namespace snippets {

class PrecisionPropagationConvertTest : public testing::Test {};

TEST_F(PrecisionPropagationConvertTest, smoke_Snippets_PrecisionPropagation_can_be_fused) {
    const std::set<std::pair<element::Type, element::Type>> precisions_set = {
        {element::u64, element::u64},
        {element::u64, element::u32},
        {element::u64, element::u16},
        {element::u64, element::u8},
        {element::u32, element::u32},
        {element::u32, element::u16},
        {element::u32, element::u8},
        {element::u16, element::u16},
        {element::u16, element::u8},
        {element::u8, element::u8},

        {element::i64, element::i64},
        {element::i64, element::i32},
        {element::i64, element::i16},
        {element::i64, element::i8},
        {element::i32, element::i32},
        {element::i32, element::i16},
        {element::i32, element::i8},
        {element::i16, element::i16},
        {element::i16, element::i8},
        {element::i8, element::i8},

        {element::f64, element::f64},
        {element::f64, element::f32},
        {element::f64, element::f16},
        {element::f32, element::f32},
        {element::f32, element::f16},
        {element::f16, element::f16},

        {element::f32, element::bf16},
        {element::bf16, element::bf16},
        {element::f32, element::i8},
        {element::f16, element::i8},
        {element::bf16, element::i8},
        {element::f32, element::u8},
        {element::f16, element::u8},
        {element::bf16, element::u8}
    };

    for (const auto& precisions : precisions_set) {
        ASSERT_TRUE(ov::snippets::pass::PropagatePrecision::can_be_fused(
            precisions.first,
            precisions.second)) << precisions.second << " can replace " << precisions.first;

        if (precisions.first == precisions.second) {
            continue;
        }

        ASSERT_FALSE(ov::snippets::pass::PropagatePrecision::can_be_fused(
            precisions.second,
            precisions.first)) << precisions.second << " can not replace " << precisions.first;
    }
}

TEST_F(PrecisionPropagationConvertTest, smoke_Snippets_PrecisionPropagation_can_not_be_fused) {
    const std::set<std::pair<element::Type, element::Type>> precisions_set = {
        {element::i64, element::f32},
        {element::i64, element::f16},
        {element::i64, element::bf16},

        {element::i32, element::f32},
        {element::i32, element::f16},
        {element::i32, element::bf16},

        {element::i16, element::f16},
        {element::i16, element::bf16},

        {element::u64, element::f32},
        {element::u64, element::f16},
        {element::u64, element::bf16},

        {element::u32, element::f32},
        {element::u32, element::f16},
        {element::u32, element::bf16},

        {element::u16, element::f16},
        {element::u16, element::bf16},

        {element::f16, element::bf16},
        {element::bf16, element::f16},

        // signed => unsigned
        {element::i64, element::u64},
        {element::i64, element::u32},
        {element::i64, element::u16},
        {element::i64, element::u8},

        {element::i32, element::u64},
        {element::i32, element::u32},
        {element::i32, element::u16},
        {element::i32, element::u8},

        {element::i16, element::u64},
        {element::i16, element::u32},
        {element::i16, element::u16},
        {element::i16, element::u8},

        {element::i8, element::u64},
        {element::i8, element::u32},
        {element::i8, element::u16},
        {element::i8, element::u8},

        // signed => unsigned
        {element::u64, element::i64},
        {element::u64, element::i32},
        {element::u64, element::i16},
        {element::u64, element::i8},

        {element::u32, element::i64},
        {element::u32, element::i32},
        {element::u32, element::i16},
        {element::u32, element::i8},

        {element::u16, element::i64},
        {element::u16, element::i32},
        {element::u16, element::i16},
        {element::u16, element::i8},

        {element::u8, element::i64},
        {element::u8, element::i32},
        {element::u8, element::i16},
        {element::u8, element::i8},
    };

    for (const auto& precisions : precisions_set) {
        ASSERT_FALSE(ov::snippets::pass::PropagatePrecision::can_be_fused(
            precisions.first,
            precisions.second)) << precisions.second << " can not replace " << precisions.first;
    }
}

TEST_F(PrecisionPropagationConvertTest, smoke_Snippets_PrecisionPropagation_can_be_removed) {
    const std::set<std::tuple<element::Type, element::Type, element::Type>> precisions_set = {
        {element::u64, element::u64, element::u64},
        {element::u32, element::u64, element::u32},
        {element::u16, element::u64, element::u16},
        {element::u8, element::u64, element::u8},
        {element::u32, element::u32, element::u32},
        {element::u16, element::u32, element::u16},
        {element::u8, element::u32, element::u8},
        {element::u16, element::u16, element::u16},
        {element::u8, element::u16, element::u8},
        {element::u8, element::u8, element::u8},

        {element::i64, element::i64, element::i64},
        {element::i32, element::i64, element::i32},
        {element::i16, element::i64, element::i16},
        {element::i8, element::i64, element::i8},
        {element::i32, element::i32, element::i32},
        {element::i16, element::i32, element::i16},
        {element::i8, element::i32, element::i8},
        {element::i16, element::i16, element::i16},
        {element::i8, element::i16, element::i8},
        {element::i8, element::i8, element::i8},

        {element::f64, element::f64, element::f64},
        {element::f32, element::f64, element::f32},
        {element::f16, element::f64, element::f16},
        {element::f32, element::f32, element::f32},
        {element::f16, element::f16, element::f16},

        {element::bf16, element::f32, element::bf16},
        {element::bf16, element::bf16, element::bf16},
    };

    for (const auto& precisions : precisions_set) {
        const auto actual_before = std::get<0>(precisions);
        const auto actual_after = std::get<1>(precisions);
        const auto required_after = std::get<2>(precisions);
        ASSERT_TRUE(ov::snippets::pass::PropagatePrecision::can_be_removed(
            actual_before,
            actual_after,
            required_after)) << "can_be_removed: " << actual_before << " => " << actual_after << " => " << required_after;

        if ((actual_before == actual_after) && (actual_before == required_after)) {
            continue;
        }
    }
}

}  // namespace snippets
}  // namespace test
}  // namespace ov