// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/layout.hpp"

#include "gtest/gtest.h"

using namespace ov;

TEST(layout, basic) {
    Layout l = "NcDHw";
    EXPECT_TRUE(layout::has_batch(l));
    EXPECT_EQ(layout::batch_idx(l), 0);
    EXPECT_TRUE(layout::has_channels(l));
    EXPECT_EQ(layout::channels_idx(l), 1);
    EXPECT_TRUE(layout::has_depth(l));
    EXPECT_EQ(layout::depth_idx(l), 2);
    EXPECT_TRUE(layout::has_height(l));
    EXPECT_EQ(layout::height_idx(l), 3);
    EXPECT_TRUE(layout::has_width(l));
    EXPECT_EQ(layout::width_idx(l), 4);
}

TEST(layout, advanced_syntax) {
    Layout l = "[batch, channels, depth, height, width]";
    EXPECT_TRUE(layout::has_batch(l));
    EXPECT_EQ(layout::batch_idx(l), 0);
    EXPECT_TRUE(layout::has_channels(l));
    EXPECT_EQ(layout::channels_idx(l), 1);
    EXPECT_TRUE(layout::has_depth(l));
    EXPECT_EQ(layout::depth_idx(l), 2);
    EXPECT_TRUE(layout::has_height(l));
    EXPECT_EQ(layout::height_idx(l), 3);
    EXPECT_TRUE(layout::has_width(l));
    EXPECT_EQ(layout::width_idx(l), 4);
    EXPECT_EQ(l, Layout("ncdhw"));

    l = "[custom1, ?, custom2]";
    EXPECT_EQ(l.get_index_by_name("CUSTOm1"), 0);
    EXPECT_EQ(l.get_index_by_name("Custom2"), 2);

    l = "[?, N , ... , channels , ?]";
    EXPECT_EQ(l, Layout("?N...C?"));
    l = "[?, N , ... , custom1 , ?]";
    EXPECT_EQ(l.get_index_by_name("Custom1"), -2);
}

TEST(layout, empty) {
    Layout l;
    EXPECT_TRUE(Layout("").empty());
    EXPECT_FALSE(layout::has_batch(l));
    EXPECT_THROW(layout::batch_idx(l), ov::AssertFailure);
    EXPECT_FALSE(layout::has_channels(l));
    EXPECT_THROW(layout::channels_idx(l), ov::AssertFailure);
    EXPECT_FALSE(layout::has_depth(l));
    EXPECT_THROW(layout::depth_idx(l), ov::AssertFailure);
    EXPECT_FALSE(layout::has_height(l));
    EXPECT_THROW(layout::height_idx(l), ov::AssertFailure);
    EXPECT_FALSE(layout::has_width(l));
    EXPECT_THROW(layout::width_idx(l), ov::AssertFailure);
}

TEST(layout, to_string) {
    std::vector<Layout> layouts = {{"NCHW"},
                                   {"[?, N, CHANNELS, ?, ?, Custom_dim_name]"},
                                   {"012?...3?456"},
                                   {"...3?456"},
                                   {"12?34..."},
                                   {"..."},
                                   Layout::scalar()};
    for (const auto& l : layouts) {
        EXPECT_EQ(l, Layout(l.to_string()));
    }
}

TEST(layout, scalar) {
    auto l = Layout::scalar();
    EXPECT_FALSE(layout::has_batch(l));
    EXPECT_THROW(layout::batch_idx(l), ov::AssertFailure);
    EXPECT_FALSE(layout::has_channels(l));
    EXPECT_THROW(layout::channels_idx(l), ov::AssertFailure);
    EXPECT_FALSE(layout::has_depth(l));
    EXPECT_THROW(layout::depth_idx(l), ov::AssertFailure);
    EXPECT_FALSE(layout::has_height(l));
    EXPECT_THROW(layout::height_idx(l), ov::AssertFailure);
    EXPECT_FALSE(layout::has_width(l));
    EXPECT_THROW(layout::width_idx(l), ov::AssertFailure);
}

TEST(layout, custom_dims) {
    Layout l = "0ac";
    EXPECT_FALSE(layout::has_batch(l));
    EXPECT_THROW(layout::batch_idx(l), ov::AssertFailure);
    EXPECT_TRUE(layout::has_channels(l));
    EXPECT_EQ(layout::channels_idx(l), 2);
    EXPECT_TRUE(l.has_name("0"));
    EXPECT_TRUE(l.has_name("A"));
    EXPECT_EQ(l.get_index_by_name("a"), 1);
}

TEST(layout, dims_unknown) {
    Layout l = "n??c";
    EXPECT_TRUE(layout::has_batch(l));
    EXPECT_EQ(layout::batch_idx(l), 0);
    EXPECT_TRUE(layout::has_channels(l));
    EXPECT_EQ(layout::channels_idx(l), 3);
    EXPECT_FALSE(l.has_name("?"));
    EXPECT_EQ(l.get_index_by_name("C"), 3);
}

TEST(layout, dims_undefined) {
    Layout l = "?n?...?c?";
    EXPECT_TRUE(layout::has_batch(l));
    EXPECT_EQ(layout::batch_idx(l), 1);
    EXPECT_TRUE(layout::has_channels(l));
    EXPECT_EQ(layout::channels_idx(l), -2);
    EXPECT_FALSE(l.has_name("?"));
}

TEST(layout, dims_valid_syntax) {
    Layout l;
    EXPECT_NO_THROW(l = "...");
    EXPECT_NO_THROW(l = "?...?");
    EXPECT_NO_THROW(l = "...?");
    EXPECT_NO_THROW(l = "?...");
    EXPECT_NO_THROW(l = "????");
    EXPECT_NO_THROW(l = "[?,?,?,?]");
    EXPECT_NO_THROW(l = "[?, ... ,?]");
    EXPECT_NO_THROW(l = "[...,?]");
    EXPECT_NO_THROW(l = "[?,...]");
}

TEST(layout, dims_wrong_syntax) {
    Layout l;
    EXPECT_THROW(l = " ", ov::AssertFailure);
    std::string invalidChars = "`~!@#$%^&*()-=+{}\"'><,|";
    for (auto c : invalidChars) {
        EXPECT_THROW(l = Layout(std::string(1, c)), ov::AssertFailure);
    }
    EXPECT_THROW(l = "....", ov::AssertFailure);
    EXPECT_THROW(l = ".nchw", ov::AssertFailure);
    EXPECT_THROW(l = "n...c...", ov::AssertFailure);
    EXPECT_THROW(l = "ncChw", ov::AssertFailure);
    EXPECT_THROW(l = "c...C", ov::AssertFailure);
    EXPECT_THROW(l = ".", ov::AssertFailure);
    EXPECT_THROW(l = "[....]", ov::AssertFailure);
    EXPECT_THROW(l = "[c, ..., n, ...]", ov::AssertFailure);
    EXPECT_THROW(l = "[c,,]", ov::AssertFailure);
    EXPECT_THROW(l = "[c", ov::AssertFailure);
    EXPECT_THROW(l = "[c]]", ov::AssertFailure);
    EXPECT_THROW(l = "[]", ov::AssertFailure);
    EXPECT_THROW(l = "[ ]", ov::AssertFailure);
    EXPECT_THROW(l = "[,]", ov::AssertFailure);
    EXPECT_THROW(l = "[...,]", ov::AssertFailure);
    EXPECT_THROW(l = "[   ]", ov::AssertFailure);
    EXPECT_THROW(l = "[?...]", ov::AssertFailure);
    EXPECT_THROW(l = "[? ...]", ov::AssertFailure);
    EXPECT_THROW(l = "[...N]", ov::AssertFailure);
    EXPECT_THROW(l = "[... N]", ov::AssertFailure);
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
    EXPECT_EQ(Layout("[Batch, ..., Channels, ? , ? ]"), Layout("N...C??"));
    EXPECT_EQ(Layout::scalar(), Layout::scalar());
    EXPECT_NE(Layout::scalar(), Layout("..."));
}

TEST(layout, dims_implicit_api) {
    EXPECT_EQ(Layout("nchw"), Layout("NCHW"));
    EXPECT_NE(Layout("nchw"), Layout("NHWC"));
    Layout l = "NCHW";
    auto l2 = l;
    auto l3 = l;
    auto l4 = std::move(l3);
    l2 = l4;
    l3 = std::move(l2);
    EXPECT_EQ(l3, l4);
}

TEST(layout, attribute_adapter) {
    Layout l = "NCHW";
    Layout l2 = "NHCW";
    AttributeAdapter<Layout> at(l);
    EXPECT_EQ(at.get(), l.to_string());
    at.set("NHCW");
    EXPECT_EQ(l, l2);
}

TEST(layout, compare_string) {
    Layout l = "HWC";
    EXPECT_EQ("[H,W,C]", l.to_string());
    Layout l2 = l.to_string().c_str();
    EXPECT_EQ(l2, l);
    EXPECT_EQ("[H,W,C]", l2.to_string());
}
