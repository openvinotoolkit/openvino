// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/op/op.hpp"
#include "openvino/pass/matcher_pass.hpp"

using namespace std;

namespace ov::test {

class OpType : public ov::op::Op {
public:
    OPENVINO_OP("OpType");
    OpType() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return nullptr;
    }
};

class OpTypeVersion : public ov::op::Op {
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
    OPENVINO_OP("OpTypeVersionParentOld", "my_version1", OpType);
    OpTypeVersionParentOld() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return nullptr;
    }
};

TEST(rtti, op_with_type) {
    auto op = OpType();
    auto type_info = op.get_type_info();
    ASSERT_EQ(type_info, OpType::get_type_info_static());
    ASSERT_EQ(strcmp(type_info.name, "OpType"), 0);
    ASSERT_EQ(strcmp(type_info.version_id, "extension"), 0);
    ASSERT_NE(type_info.parent, nullptr);
    ASSERT_EQ(*type_info.parent, ov::op::Op::get_type_info_static());
}

TEST(rtti, op_with_type_version) {
    auto op = OpTypeVersion();
    auto type_info = op.get_type_info();
    ASSERT_EQ(type_info, OpTypeVersion::get_type_info_static());
    ASSERT_EQ(strcmp(type_info.name, "OpTypeVersion"), 0);
    ASSERT_EQ(strcmp(type_info.version_id, "my_version"), 0);
    ASSERT_NE(type_info.parent, nullptr);
    ASSERT_EQ(*type_info.parent, ov::op::Op::get_type_info_static());
}

TEST(rtti, op_with_type_version_parent) {
    auto op = OpTypeVersionParent();
    auto type_info = op.get_type_info();
    ASSERT_EQ(type_info, OpTypeVersionParent::get_type_info_static());
    ASSERT_EQ(strcmp(type_info.name, "OpTypeVersionParent"), 0);
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
    ASSERT_NE(type_info.parent, nullptr);
    ASSERT_EQ(*type_info.parent, OpType::get_type_info_static());
}

#if !defined(__ANDROID__) && !defined(ANDROID)

class IncompleteRtti : public pass::MatcherPass {
public:
    OPENVINO_RTTI("IncompleteRtti", "rtti_test");
};

class DerivedIncompleteRtti : public IncompleteRtti {
public:
    OPENVINO_RTTI("DerivedIncompleteRtti", "rtti_test", IncompleteRtti);
};

// Assert backward compatibility of RTTI definition without parent but casted with as_type or as_type_ptr pointer work.
TEST(rtti, assert_casting_without_parent) {
    {
        IncompleteRtti incomplete;
        DerivedIncompleteRtti derived;

        auto pass_A = as_type<pass::MatcherPass>(&incomplete);
        auto pass_B = as_type<pass::MatcherPass>(&derived);
        auto pass_C = as_type<IncompleteRtti>(&derived);

        EXPECT_NE(nullptr, pass_A);
        EXPECT_NE(nullptr, pass_B);
        EXPECT_NE(nullptr, pass_C);

        EXPECT_NE(nullptr, as_type<IncompleteRtti>(pass_A));
        EXPECT_NE(nullptr, as_type<IncompleteRtti>(pass_B));
        EXPECT_NE(nullptr, as_type<DerivedIncompleteRtti>(pass_B));
        EXPECT_NE(nullptr, as_type<DerivedIncompleteRtti>(pass_C));
    }
    {
        auto incomplete = std::make_shared<IncompleteRtti>();
        auto derived = std::make_shared<DerivedIncompleteRtti>();

        auto pass_A = as_type_ptr<pass::MatcherPass>(incomplete);
        auto pass_B = as_type_ptr<pass::MatcherPass>(derived);
        auto pass_C = as_type_ptr<IncompleteRtti>(derived);

        EXPECT_NE(nullptr, pass_A);
        EXPECT_NE(nullptr, pass_B);
        EXPECT_NE(nullptr, pass_C);

        EXPECT_NE(nullptr, as_type_ptr<IncompleteRtti>(pass_A));
        EXPECT_NE(nullptr, as_type_ptr<IncompleteRtti>(pass_B));
        EXPECT_NE(nullptr, as_type_ptr<DerivedIncompleteRtti>(pass_B));
        EXPECT_NE(nullptr, as_type_ptr<DerivedIncompleteRtti>(pass_C));
    }
}
#endif  // ANDROID
}  // namespace ov::test
