// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, constant_op) {
    ov::test::NodeBuilder::opset().insert<ov::op::v0::Constant>();
    vector<float> data{5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f};
    auto k = make_shared<op::v0::Constant>(element::f32, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    EXPECT_EQ(k->get_byte_size(), g_k->get_byte_size());
    vector<float> g_data = g_k->get_vector<float>();
    EXPECT_EQ(data, g_data);
}

TEST(attributes, constant_op_different_elements) {
    vector<int64_t> data{5, 4, 3, 2, 1, 0};
    auto k = make_shared<op::v0::Constant>(element::i64, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<int64_t> g_data = g_k->get_vector<int64_t>();
    EXPECT_EQ(data, g_data);
    ASSERT_FALSE(g_k->get_all_data_elements_bitwise_identical());
}

TEST(attributes, constant_op_sub_byte_deserialize_strides) {
    // Default-constructed Constant loaded via visit_attributes (e.g. deserialization) must get its
    // strides cache populated for a representable sub-byte shape, not left empty.
    auto k = make_shared<op::v0::Constant>(element::u4, Shape{2, 100});
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_strides(), g_k->get_strides());
    EXPECT_EQ(g_k->get_strides(), ov::Strides({50, 1}));
}

TEST(attributes, constant_op_sub_byte_reload_strides_on_shape_change) {
    // Re-applying visit_attributes to an already populated Constant with a new shape must refresh
    // the cached strides instead of retaining the stale value for the previous shape (which would
    // make get_tensor_view()/get_strides() report an out-of-bounds layout for the new buffer).
    auto k = make_shared<op::v0::Constant>(element::u4, Shape{2, 100});
    ASSERT_EQ(k->get_strides(), ov::Strides({50, 1}));

    auto other = make_shared<op::v0::Constant>(element::u4, Shape{100, 2});
    NodeBuilder builder(other);
    k->visit_attributes(builder);
    k->validate_and_infer_types();

    EXPECT_EQ(k->get_shape(), (Shape{100, 2}));
    EXPECT_EQ(k->get_strides(), ov::Strides({1, 1}));
}

TEST(attributes, constant_op_identical_elements) {
    vector<int64_t> data{5, 5, 5, 5, 5, 5};
    auto k = make_shared<op::v0::Constant>(element::i64, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<int64_t> g_data = g_k->get_vector<int64_t>();
    EXPECT_EQ(data, g_data);
    ASSERT_TRUE(g_k->get_all_data_elements_bitwise_identical());
}

TEST(attributes, constant_op_from_host_tensor_different_elements) {
    vector<int64_t> data{5, 4, 3, 2, 1, 0};
    auto tensor = ov::Tensor(element::i64, Shape{2, 3}, &data[0]);
    auto k = make_shared<op::v0::Constant>(tensor);
    ASSERT_FALSE(k->get_all_data_elements_bitwise_identical());
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<int64_t> g_data = g_k->get_vector<int64_t>();
    EXPECT_EQ(data, g_data);
    ASSERT_FALSE(g_k->get_all_data_elements_bitwise_identical());
}

TEST(attributes, constant_op_from_host_tensor_identical_elements) {
    vector<int64_t> data{5, 5, 5, 5, 5, 5};
    auto tensor = ov::Tensor(element::i64, Shape{2, 3}, &data[0]);
    auto k = make_shared<op::v0::Constant>(tensor);
    ASSERT_TRUE(k->get_all_data_elements_bitwise_identical());
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<int64_t> g_data = g_k->get_vector<int64_t>();
    EXPECT_EQ(data, g_data);
    ASSERT_TRUE(g_k->get_all_data_elements_bitwise_identical());
}

TEST(attributes, constant_op_string) {
    vector<std::string> data{"abc", "de fc qq", "", "123 abc", "0112 3     ", "&&&"};
    auto k = make_shared<op::v0::Constant>(element::string, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<std::string> g_data = g_k->get_vector<std::string>();
    EXPECT_EQ(data, g_data);
    ASSERT_FALSE(g_k->get_all_data_elements_bitwise_identical());
}

TEST(attributes, constant_op_identical_elements_string) {
    vector<std::string> data{"abc edfg", "abc edfg", "abc edfg", "abc edfg", "abc edfg", "abc edfg"};
    auto k = make_shared<op::v0::Constant>(element::string, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<std::string> g_data = g_k->get_vector<std::string>();
    EXPECT_EQ(data, g_data);
    ASSERT_TRUE(g_k->get_all_data_elements_bitwise_identical());
}

TEST(attributes, constant_op_from_host_tensor_different_elements_string) {
    vector<std::string> data{"abc", "de fc qq", "", "123 abc", "0112 3     ", "&&&"};
    auto tensor = ov::Tensor(element::string, Shape{2, 3}, &data[0]);
    auto k = make_shared<op::v0::Constant>(tensor);
    ASSERT_FALSE(k->get_all_data_elements_bitwise_identical());
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<std::string> g_data = g_k->get_vector<std::string>();
    EXPECT_EQ(data, g_data);
    ASSERT_FALSE(g_k->get_all_data_elements_bitwise_identical());
}

TEST(attributes, constant_op_from_host_tensor_identical_elements_string) {
    vector<std::string> data{"abc edfg", "abc edfg", "abc edfg", "abc edfg", "abc edfg", "abc edfg"};
    auto tensor = ov::Tensor(element::string, Shape{2, 3}, &data[0]);
    auto k = make_shared<op::v0::Constant>(tensor);
    ASSERT_TRUE(k->get_all_data_elements_bitwise_identical());
    NodeBuilder builder(k);
    auto g_k = ov::as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<std::string> g_data = g_k->get_vector<std::string>();
    EXPECT_EQ(data, g_data);
    ASSERT_TRUE(g_k->get_all_data_elements_bitwise_identical());
}
