// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shared_any.hpp"

#include <string>

#include "gtest/gtest.h"

TEST(op, any_std_string) {
    using namespace ov;
    auto any = SharedAny{std::string{"My string"}};
    ASSERT_TRUE(any.is<std::string>());
    EXPECT_EQ(any.as<std::string>(), "My string");
}

TEST(op, any_int64_t) {
    using namespace ov;
    auto any = SharedAny{27ll};
    ASSERT_TRUE(any.is<int64_t>());
    EXPECT_FALSE(any.is<std::string>());
    EXPECT_EQ(any.as<int64_t>(), 27);
}

struct Ship {
    std::string name;
    int16_t x;
    int16_t y;
};

TEST(op, any_ship) {
    using namespace ov;
    {
        auto any = SharedAny{Ship{"Lollipop", 3, 4}};
        ASSERT_TRUE(any.is<Ship>());
        Ship& ship = any.as<Ship>();
        EXPECT_EQ(ship.name, "Lollipop");
        EXPECT_EQ(ship.x, 3);
        EXPECT_EQ(ship.y, 4);
    }
    {
        auto any = SharedAny::make<Ship>("Lollipop", int16_t(3), int16_t(4));
        ASSERT_TRUE(any.is<Ship>());
        Ship& ship = any.as<Ship>();
        EXPECT_EQ(ship.name, "Lollipop");
        EXPECT_EQ(ship.x, 3);
        EXPECT_EQ(ship.y, 4);
    }
    {
        auto any = SharedAny::make<Ship>("Lollipop", int16_t(3), int16_t(4));
        ASSERT_TRUE(any.is<Ship>());
        Ship ship = any;
        EXPECT_EQ(ship.name, "Lollipop");
        EXPECT_EQ(ship.x, 3);
        EXPECT_EQ(ship.y, 4);
    }
}
