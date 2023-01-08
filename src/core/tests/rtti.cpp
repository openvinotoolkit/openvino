// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/node.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

class OpType : public ngraph::op::Op {
public:
    OPENVINO_OP("OpType");
    OpType() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return nullptr;
    }
};

class OpTypeVersion : public ngraph::op::Op {
public:
    OPENVINO_OP("OpTypeVersion", "my_version");
    OpTypeVersion() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return nullptr;
    }
};

class OpTypeVersionParent : public OpType {
public:
    OPENVINO_OP("OpTypeVersionParent", "my_version", OpType);
    OpTypeVersionParent() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return nullptr;
    }
};

class OpTypeVersionParentOld : public OpType {
public:
    OPENVINO_OP("OpTypeVersionParentOld", "my_version1", OpType, 1);
    OpTypeVersionParentOld() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return nullptr;
    }
};

OPENVINO_SUPPRESS_DEPRECATED_START
TEST(rtti, op_with_type) {
    auto op = OpType();
    auto type_info = op.get_type_info();
    ASSERT_EQ(type_info, OpType::get_type_info_static());
    ASSERT_EQ(strcmp(type_info.name, "OpType"), 0);
    ASSERT_EQ(type_info.version, 0);
    ASSERT_EQ(strcmp(type_info.version_id, "extension"), 0);
    ASSERT_NE(type_info.parent, nullptr);
    ASSERT_EQ(*type_info.parent, ngraph::op::Op::get_type_info_static());
}

TEST(rtti, op_with_type_version) {
    auto op = OpTypeVersion();
    auto type_info = op.get_type_info();
    ASSERT_EQ(type_info, OpTypeVersion::get_type_info_static());
    ASSERT_EQ(strcmp(type_info.name, "OpTypeVersion"), 0);
    ASSERT_EQ(type_info.version, 0);
    ASSERT_EQ(strcmp(type_info.version_id, "my_version"), 0);
    ASSERT_NE(type_info.parent, nullptr);
    ASSERT_EQ(*type_info.parent, ngraph::op::Op::get_type_info_static());
}

TEST(rtti, op_with_type_version_parent) {
    auto op = OpTypeVersionParent();
    auto type_info = op.get_type_info();
    ASSERT_EQ(type_info, OpTypeVersionParent::get_type_info_static());
    ASSERT_EQ(strcmp(type_info.name, "OpTypeVersionParent"), 0);
    ASSERT_EQ(type_info.version, 0);
    ASSERT_EQ(strcmp(type_info.version_id, "my_version"), 0);
    ASSERT_NE(type_info.parent, nullptr);
    ASSERT_EQ(*type_info.parent, OpType::get_type_info_static());
}

TEST(rtti, op_with_type_version_parent_old) {
    auto op = OpTypeVersionParentOld();
    auto type_info = op.get_type_info();
    ASSERT_EQ(type_info, OpTypeVersionParentOld::get_type_info_static());
    ASSERT_EQ(strcmp(type_info.name, "OpTypeVersionParentOld"), 0);
    ASSERT_EQ(strcmp(type_info.version_id, "my_version1"), 0);
    ASSERT_EQ(type_info.version, 1);
    ASSERT_NE(type_info.parent, nullptr);
    ASSERT_EQ(*type_info.parent, OpType::get_type_info_static());
}
