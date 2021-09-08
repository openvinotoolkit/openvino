// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/layout.hpp"

#include "gtest/gtest.h"

using namespace ov;

TEST(layout, basic) {
    auto l = Layout("NcDHw");
    EXPECT_TRUE(layouts::has_batch(l));
    EXPECT_EQ(layouts::batch(l), 0);
    EXPECT_TRUE(layouts::has_channels(l));
    EXPECT_EQ(layouts::channels(l), 1);
    EXPECT_TRUE(layouts::has_depth(l));
    EXPECT_EQ(layouts::depth(l), 2);
    EXPECT_TRUE(layouts::has_height(l));
    EXPECT_EQ(layouts::height(l), 3);
    EXPECT_TRUE(layouts::has_width(l));
    EXPECT_EQ(layouts::width(l), 4);
    EXPECT_FALSE(l.is_scalar());
}

TEST(layout, empty) {
    auto l = Layout();
    EXPECT_FALSE(layouts::has_batch(l));
    EXPECT_ANY_THROW(layouts::batch(l));
    EXPECT_FALSE(layouts::has_channels(l));
    EXPECT_ANY_THROW(layouts::channels(l));
    EXPECT_FALSE(layouts::has_depth(l));
    EXPECT_ANY_THROW(layouts::depth(l));
    EXPECT_FALSE(layouts::has_height(l));
    EXPECT_ANY_THROW(layouts::height(l));
    EXPECT_FALSE(layouts::has_width(l));
    EXPECT_ANY_THROW(layouts::width(l));
    EXPECT_FALSE(l.is_scalar());
}

TEST(layout, scalar) {
    auto l = Layout::scalar();
    EXPECT_TRUE(l.is_scalar());
    EXPECT_FALSE(layouts::has_batch(l));
    EXPECT_ANY_THROW(layouts::batch(l));
    EXPECT_FALSE(layouts::has_channels(l));
    EXPECT_ANY_THROW(layouts::channels(l));
    EXPECT_FALSE(layouts::has_depth(l));
    EXPECT_ANY_THROW(layouts::depth(l));
    EXPECT_FALSE(layouts::has_height(l));
    EXPECT_ANY_THROW(layouts::height(l));
    EXPECT_FALSE(layouts::has_width(l));
    EXPECT_ANY_THROW(layouts::width(l));
}

TEST(layout, custom_dims) {
    auto l = Layout("0ac");
    EXPECT_FALSE(layouts::has_batch(l));
    EXPECT_ANY_THROW(layouts::batch(l));
    EXPECT_TRUE(layouts::has_channels(l));
    EXPECT_EQ(layouts::channels(l), 2);
    EXPECT_TRUE(l.has_name("0"));
    EXPECT_TRUE(l.has_name("A"));
    EXPECT_EQ(l.get_index_by_name("A"), 1);
}

TEST(layout, dims_unknown) {
    auto l = Layout("n??c");
    EXPECT_TRUE(layouts::has_batch(l));
    EXPECT_EQ(layouts::batch(l), 0);
    EXPECT_TRUE(layouts::has_channels(l));
    EXPECT_EQ(layouts::channels(l), 3);
    EXPECT_FALSE(l.has_name("?"));
    EXPECT_EQ(l.get_index_by_name("C"), 3);
}

TEST(layout, dims_undefined) {
    auto l = Layout("?n?...?c?");
    EXPECT_TRUE(layouts::has_batch(l));
    EXPECT_EQ(layouts::batch(l), 1);
    EXPECT_TRUE(layouts::has_channels(l));
    EXPECT_EQ(layouts::channels(l), -2);
    EXPECT_FALSE(l.has_name("?"));
}

TEST(layout, dims_valid_syntax) {
    auto l = Layout();
    EXPECT_NO_THROW(l = Layout("..."));
    EXPECT_NO_THROW(l = Layout("?...?"));
    EXPECT_NO_THROW(l = Layout("????"));
}

TEST(layout, dims_wrong_syntax) {
    auto l = Layout();
    EXPECT_ANY_THROW(l = Layout(""));
    std::string invalidChars = "`~!@#$%^&*()-_=+{}\"'><,|";
    for (auto c : invalidChars) {
        EXPECT_THROW(l = Layout(std::string(1, c)), ov::CheckFailure);
    }
    EXPECT_THROW(l = Layout("...."), ov::CheckFailure);
    EXPECT_THROW(l = Layout(".nchw"), ov::CheckFailure);
    EXPECT_THROW(l = Layout("n...c..."), ov::CheckFailure);
    EXPECT_THROW(l = Layout("ncChw"), ov::CheckFailure);
    EXPECT_THROW(l = Layout("c...C"), ov::CheckFailure);
    EXPECT_THROW(l = Layout("."), ov::CheckFailure);
}

TEST(layout, dims_implicit_api) {
    EXPECT_EQ(Layout("nchw"), Layout("NCHW"));
    EXPECT_NE(Layout("nchw"), Layout("NHWC"));
    auto l = Layout("NCHW");
    auto l2 = l;
    auto l3 = std::move(l);
    l2 = l3;
    l3 = std::move(l2);
}