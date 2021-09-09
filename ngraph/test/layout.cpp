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

TEST(layout, advanced_syntax) {
    auto l = Layout("[batch, channels, depth, height, width]");
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
    EXPECT_EQ(l, Layout("ncdhw"));

    l = Layout("[custom1, ?, custom2]");
    EXPECT_EQ(l.get_index_by_name("CUSTOm1"), 0);
    EXPECT_EQ(l.get_name_by_index(1), std::string());
    EXPECT_EQ(l.get_index_by_name("Custom2"), 2);

    l = Layout("[?, N , ... , channels , ?]");
    EXPECT_EQ(l, Layout("?N...C?"));
    l = Layout("[?, N , ... , custom1 , ?]");
    EXPECT_EQ(l.get_index_by_name("Custom1"), -2);
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

TEST(layout, predefined_names) {
    using namespace layouts;
    auto l = Layout("[?, ?, ?, ?, ?, Custom_dim_name]");
    set_batch(l, 0);
    set_channels(l, 1);
    set_depth(l, 2);
    set_height(l, 3);
    set_width(l, 4);
    EXPECT_EQ(l.get_index_by_name(predefined_name(PredefinedDim::BATCH)), batch(l));
    EXPECT_EQ(l.get_index_by_name(predefined_name(PredefinedDim::CHANNELS)), channels(l));
    EXPECT_EQ(l.get_index_by_name(predefined_name(PredefinedDim::DEPTH)), depth(l));
    EXPECT_EQ(l.get_index_by_name(predefined_name(PredefinedDim::WIDTH)), width(l));
    EXPECT_EQ(l.get_index_by_name(predefined_name(PredefinedDim::HEIGHT)), height(l));
    EXPECT_EQ(to_predefined_dim(l.get_name_by_index(5)), PredefinedDim::UNDEFINED);
    EXPECT_EQ(layouts::predefined_name(to_predefined_dim(l.get_name_by_index(5))), std::string());
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
    EXPECT_NO_THROW(l = Layout("...?"));
    EXPECT_NO_THROW(l = Layout("?..."));
    EXPECT_NO_THROW(l = Layout("????"));
    EXPECT_NO_THROW(l = Layout("[?,?,?,?]"));
    EXPECT_NO_THROW(l = Layout("[?, ... ,?]"));
    EXPECT_NO_THROW(l = Layout("[...,?]"));
    EXPECT_NO_THROW(l = Layout("[?,...]"));
}

TEST(layout, dims_wrong_syntax) {
    auto l = Layout();
    EXPECT_ANY_THROW(l = Layout(""));
    EXPECT_ANY_THROW(l = Layout(" "));
    std::string invalidChars = "`~!@#$%^&*()-=+{}\"'><,|";
    for (auto c : invalidChars) {
        EXPECT_THROW(l = Layout(std::string(1, c)), ov::AssertFailure);
    }
    EXPECT_THROW(l = Layout("...."), ov::AssertFailure);
    EXPECT_THROW(l = Layout(".nchw"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("n...c..."), ov::AssertFailure);
    EXPECT_THROW(l = Layout("ncChw"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("c...C"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("."), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[....]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[c, ..., n, ...]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[c,,]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[c"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[c]]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[ ]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[,]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[...,]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[   ]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[?...]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[? ...]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[...N]"), ov::AssertFailure);
    EXPECT_THROW(l = Layout("[... N]"), ov::AssertFailure);
}

TEST(layout, get_name_by_index) {
    auto l1 = Layout("n?hw");
    EXPECT_EQ(layouts::to_predefined_dim(l1.get_name_by_index(0)), layouts::PredefinedDim::BATCH);
    EXPECT_TRUE(l1.get_name_by_index(1).empty());
    EXPECT_EQ(layouts::to_predefined_dim(l1.get_name_by_index(2)), layouts::PredefinedDim::HEIGHT);
    EXPECT_EQ(layouts::to_predefined_dim(l1.get_name_by_index(3)), layouts::PredefinedDim::WIDTH);
    EXPECT_THROW(l1.get_name_by_index(-1), ov::AssertFailure);
    EXPECT_THROW(l1.get_name_by_index(5), ov::AssertFailure);

    l1 = Layout("?n...c??");
    EXPECT_TRUE(l1.get_name_by_index(0).empty());
    EXPECT_TRUE(l1.get_name_by_index(-1).empty());
    EXPECT_TRUE(l1.get_name_by_index(-2).empty());
    EXPECT_EQ(layouts::to_predefined_dim(l1.get_name_by_index(1)), layouts::PredefinedDim::BATCH);
    EXPECT_EQ(layouts::to_predefined_dim(l1.get_name_by_index(-3)), layouts::PredefinedDim::CHANNELS);
    EXPECT_THROW(l1.get_name_by_index(-4), ov::AssertFailure);
    EXPECT_THROW(l1.get_name_by_index(2), ov::AssertFailure);

    l1 = Layout::scalar();
    EXPECT_THROW(l1.get_name_by_index(0), ov::AssertFailure);
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
    l.set_name_for_index("N", 1);  // becomes "?N..."
    EXPECT_TRUE(l.has_name("n"));
    l.set_name_for_index("C", 3);  // becomes "?N?C..."
    EXPECT_TRUE(layouts::has_channels(l));
    EXPECT_EQ(layouts::channels(l), 3);
    EXPECT_TRUE(l.rank().is_dynamic());
    EXPECT_EQ(l.rank().size_left(), 4);
    l.set_name_for_index("H", -2);  // becomes "?N?C...H?"
    EXPECT_EQ(l.rank().size_right(), 2);
    EXPECT_EQ(layouts::height(l), -2);

    l.set_name_for_index("1", 2);  // becomes "?N1C...H?"
    EXPECT_EQ(l.rank().size_left(), 4);
    EXPECT_TRUE(l.has_name("1"));
    EXPECT_EQ(l.get_index_by_name("1"), 2);

    l.set_name_for_index("q", -5);  // becomes "?N1C...Q??H?"
    EXPECT_EQ(l.rank().size_right(), 5);
    EXPECT_TRUE(l.has_name("q"));
    EXPECT_EQ(l.get_index_by_name("Q"), -5);

    l.set_name_for_index("z", -4);  // becomes "?N1C...QZ?H?"
    EXPECT_EQ(l.rank().size_right(), 5);
    EXPECT_TRUE(l.has_name("Z"));
    EXPECT_EQ(l.get_index_by_name("z"), -4);
}

TEST(layout, invalid_set_name_for_index) {
    auto l = Layout("n?hw");
    EXPECT_THROW(l.set_name_for_index("w", 1), ov::AssertFailure);
    EXPECT_THROW(l.set_name_for_index("c", 0), ov::AssertFailure);
    EXPECT_THROW(l.set_name_for_index("c", 4), ov::AssertFailure);
    EXPECT_THROW(l.set_name_for_index("c", -1), ov::AssertFailure);
    EXPECT_THROW(l.set_name_for_index("___", 1), ov::AssertFailure);
    EXPECT_THROW(l.set_name_for_index("?", 1), ov::AssertFailure);
    EXPECT_THROW(l.set_name_for_index("...", 1), ov::AssertFailure);
    EXPECT_THROW(l.set_name_for_index("[]", 1), ov::AssertFailure);
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
    auto l3 = l;
    auto l4 = std::move(l3);
    l2 = l4;
    l3 = std::move(l2);
    EXPECT_EQ(l3, l4);
}

TEST(layout, attribute_adapter) {
    auto l = Layout("NCHW");
    auto l2 = Layout("NHCW");
    AttributeAdapter<Layout> at(l);
    EXPECT_EQ(at.get(), l.to_string());
    at.set("NHCW");
    EXPECT_EQ(l, l2);
}