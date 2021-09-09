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
    EXPECT_EQ(l.get_index_by_name("a"), 1);
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

TEST(layout, set_name_for_index) {
    auto l = Layout("n?hw");
    EXPECT_FALSE(layouts::has_channels(l));
    l.set_name_for_index("C", 1);
    EXPECT_TRUE(layouts::has_channels(l));
    EXPECT_NO_THROW(l.set_name_for_index("c", 1));
    EXPECT_NO_THROW(l.set_name_for_index("N", 0));

    l = Layout("...");
    EXPECT_TRUE(l.rank().is_dynamic());
    EXPECT_EQ(l.rank().size_left(), 0);
    l.set_name_for_index("N", 1);          // becomes "?N..."
    EXPECT_TRUE(l.has_name("n"));
    l.set_name_for_index("C", 3);          // becomes "?N?C..."
    EXPECT_TRUE(layouts::has_channels(l));
    EXPECT_EQ(layouts::channels(l), 3);
    EXPECT_TRUE(l.rank().is_dynamic());
    EXPECT_EQ(l.rank().size_left(), 4);
    l.set_name_for_index("H", -2);         // becomes "?N?C...H?"
    EXPECT_EQ(l.rank().size_right(), 2);
    EXPECT_EQ(layouts::height(l), -2);

    l.set_name_for_index("1", 2);          // becomes "?N1C...H?"
    EXPECT_EQ(l.rank().size_left(), 4);
    EXPECT_TRUE(l.has_name("1"));
    EXPECT_EQ(l.get_index_by_name("1"), 2);

    l.set_name_for_index("q", -5);          // becomes "?N1C...Q??H?"
    EXPECT_EQ(l.rank().size_right(), 5);
    EXPECT_TRUE(l.has_name("q"));
    EXPECT_EQ(l.get_index_by_name("Q"), -5);

    l.set_name_for_index("z", -4);          // becomes "?N1C...QZ?H?"
    EXPECT_EQ(l.rank().size_right(), 5);
    EXPECT_TRUE(l.has_name("Z"));
    EXPECT_EQ(l.get_index_by_name("z"), -4);
}

TEST(layout, invalid_set_name_for_index) {
    auto l = Layout("n?hw");
    EXPECT_THROW(l.set_name_for_index("w", 1), ov::CheckFailure);
    EXPECT_THROW(l.set_name_for_index("c", 0), ov::CheckFailure);
    EXPECT_THROW(l.set_name_for_index("c", 4), ov::CheckFailure);
    EXPECT_THROW(l.set_name_for_index("c", -1), ov::CheckFailure);
    EXPECT_THROW(l.set_name_for_index("___", 1), ov::CheckFailure);
    EXPECT_THROW(l.set_name_for_index("?", 1), ov::CheckFailure);
    EXPECT_THROW(l.set_name_for_index("...", 1), ov::CheckFailure);
    EXPECT_THROW(l.set_name_for_index("[]", 1), ov::CheckFailure);
}

TEST(layout, layout_equals) {
    EXPECT_EQ(Layout("nchw"), Layout("NCHW"));
    EXPECT_NE(Layout("nc?hw"), Layout("NC...HW"));
    EXPECT_NE(Layout("nc"), Layout("NC..."));
    EXPECT_NE(Layout("n???"), Layout("NCHW"));
    EXPECT_NE(Layout("nchw"), Layout("n???"));
    EXPECT_NE(Layout("?n..."), Layout("...n?"));
    EXPECT_EQ(Layout("0A?...HWC"), Layout("0a?...hwc"));
    EXPECT_NE(Layout("?..."), Layout("?"));
    EXPECT_NE(Layout("...?"), Layout("..."));
    EXPECT_EQ(Layout::scalar(), Layout::scalar());
    EXPECT_NE(Layout::scalar(), Layout("..."));
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