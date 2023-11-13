// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <gtest/gtest.h>
#include <ie_blob.h>
#include "low_precision/layer_transformation.hpp"
#include "low_precision/network_helper.hpp"
#include "ov_models/builders.hpp"

using namespace ov;

TEST(smoke_LPT_DataPrecision, check) {
    using namespace ov::pass::low_precision;

    const std::vector<element::Type> type_items = {
        element::i4,
        element::u4,
        element::i8,
        element::u8,
        element::i16,
        element::u16,
        element::i32,
        element::u32
    };

    const std::vector<levels> level_items = {
        int4,
        int4_narrow_range,
        int8,
        int8_narrow_range,
        int16,
        int16_narrow_range,
        int32,
        int32_narrow_range
    };

    const std::map<element::Type, std::set<levels>> items = {
        {element::i4, {levels::int4, levels::int4_narrow_range}},
        {element::u4, {levels::int4, levels::int4_narrow_range}},
        {element::i8, {levels::int8, levels::int8_narrow_range}},
        {element::u8, {levels::int8, levels::int8_narrow_range}},
        {element::i16, {levels::int16, levels::int16_narrow_range}},
        {element::u16, {levels::int16, levels::int16_narrow_range}},
        {element::i32, {levels::int32, levels::int32_narrow_range}},
        {element::u32, {levels::int32, levels::int32_narrow_range}},
    };
    for (const auto type_item : type_items) {
        for (const auto level_item : level_items) {
            const auto& levels = items.find(type_item)->second;
            if (levels.find(level_item) == levels.end()) {
                ASSERT_FALSE(DataPrecision::check(type_item, level_item));
            } else {
                ASSERT_TRUE(DataPrecision::check(type_item, level_item));
            }
        }
    }
}
