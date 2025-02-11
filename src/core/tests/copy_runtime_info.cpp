// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

using namespace ov;
using namespace std;

class TestAttributeNoCopyable : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("TestAttributeNoCopyable", "0", RuntimeAttribute);
    TestAttributeNoCopyable() = default;
    bool is_copyable() const override {
        return false;
    }

    static void set(std::shared_ptr<Node> node) {
        auto& rt_info = node->get_rt_info();
        rt_info[TestAttributeNoCopyable::get_type_info_static()] = TestAttributeNoCopyable();
    }

    static bool exists_in(std::shared_ptr<Node> node) {
        const auto& rt_info = node->get_rt_info();
        return rt_info.count(TestAttributeNoCopyable::get_type_info_static());
    }
};

class TestAttributeCopyable : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("TestAttributeCopyable", "0", RuntimeAttribute);
    TestAttributeCopyable() = default;

    static void set(std::shared_ptr<Node> node) {
        auto& rt_info = node->get_rt_info();
        rt_info[TestAttributeCopyable::get_type_info_static()] = TestAttributeCopyable();
    }

    static bool exists_in(std::shared_ptr<Node> node) {
        const auto& rt_info = node->get_rt_info();
        return rt_info.count(TestAttributeCopyable::get_type_info_static());
    }
};

class TestAttributeMergable : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("TestAttributeMergable", "0", RuntimeAttribute);
    TestAttributeMergable() = default;

    static void set(std::shared_ptr<Node> node) {
        auto& rt_info = node->get_rt_info();
        rt_info[TestAttributeMergable::get_type_info_static()] = TestAttributeMergable();
    }

    static bool exists_in(std::shared_ptr<Node> node) {
        const auto& rt_info = node->get_rt_info();
        return rt_info.count(TestAttributeMergable::get_type_info_static());
    }

    ov::Any merge(const ov::NodeVector& nodes) const override {
        return {TestAttributeMergable()};
    }
};

TEST(copy_runtime_info, node_to_node_1) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto b = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    TestAttributeCopyable::set(a);
    TestAttributeNoCopyable::set(b);

    ov::copy_runtime_info(a, b);

    ASSERT_TRUE(TestAttributeCopyable::exists_in(a));
    ASSERT_TRUE(TestAttributeCopyable::exists_in(b));

    ASSERT_TRUE(TestAttributeNoCopyable::exists_in(b));

    copy_runtime_info(b, b);
    ASSERT_TRUE(TestAttributeCopyable::exists_in(b));
    ASSERT_TRUE(TestAttributeNoCopyable::exists_in(b));
}

TEST(copy_runtime_info, node_to_node_2) {
    auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto b = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    TestAttributeCopyable::set(a);
    TestAttributeNoCopyable::set(a);

    copy_runtime_info(a, b);

    ASSERT_TRUE(TestAttributeCopyable::exists_in(a));
    ASSERT_TRUE(TestAttributeNoCopyable::exists_in(a));

    ASSERT_TRUE(TestAttributeCopyable::exists_in(b));
    ASSERT_FALSE(TestAttributeNoCopyable::exists_in(b));
}

TEST(copy_runtime_info, node_to_nodes) {
    auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto b = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto c = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    TestAttributeCopyable::set(a);
    TestAttributeNoCopyable::set(b);
    TestAttributeNoCopyable::set(c);

    copy_runtime_info(a, {b, c});

    ASSERT_TRUE(TestAttributeCopyable::exists_in(a));
    ASSERT_TRUE(TestAttributeCopyable::exists_in(b));
    ASSERT_TRUE(TestAttributeCopyable::exists_in(c));

    ASSERT_FALSE(TestAttributeNoCopyable::exists_in(a));
    ASSERT_TRUE(TestAttributeNoCopyable::exists_in(b));
    ASSERT_TRUE(TestAttributeNoCopyable::exists_in(c));
}

TEST(copy_runtime_info, nodes_to_node_1) {
    auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto b = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto c = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    TestAttributeCopyable::set(a);
    TestAttributeNoCopyable::set(a);

    TestAttributeCopyable::set(b);
    TestAttributeNoCopyable::set(b);

    copy_runtime_info({a, b}, c);

    ASSERT_FALSE(TestAttributeCopyable::exists_in(c));
    ASSERT_FALSE(TestAttributeNoCopyable::exists_in(c));
}

TEST(copy_runtime_info, nodes_to_node_2) {
    auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto b = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto c = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    TestAttributeMergable::set(a);
    TestAttributeMergable::set(b);
    TestAttributeNoCopyable::set(c);

    copy_runtime_info({a, b}, c);

    ASSERT_TRUE(TestAttributeMergable::exists_in(c));
    ASSERT_TRUE(TestAttributeNoCopyable::exists_in(c));
}

TEST(copy_runtime_info, nodes_to_node_3) {
    auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto b = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    TestAttributeCopyable::set(a);
    TestAttributeNoCopyable::set(b);

    copy_runtime_info({a, b}, b);

    ASSERT_TRUE(TestAttributeCopyable::exists_in(b));
    ASSERT_TRUE(TestAttributeNoCopyable::exists_in(b));
}

TEST(copy_runtime_info, replace_output_update_name) {
    auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto b = make_shared<op::v0::Relu>(a);
    auto c = make_shared<op::v0::Relu>(b);
    auto d = make_shared<op::v0::Relu>(c);

    TestAttributeMergable::set(b);
    TestAttributeMergable::set(c);

    TestAttributeCopyable::set(c);

    TestAttributeNoCopyable::set(b);
    TestAttributeNoCopyable::set(c);

    // performs copy_runtime_info like copy_runtime_info({b, c}, b);
    ov::replace_output_update_name(c->output(0), b->output(0));

    ASSERT_TRUE(TestAttributeCopyable::exists_in(b));
    ASSERT_TRUE(TestAttributeMergable::exists_in(b));
    ASSERT_TRUE(TestAttributeNoCopyable::exists_in(b));
}
