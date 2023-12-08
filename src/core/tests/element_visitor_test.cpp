// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "element_visitor.hpp"

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"

using namespace testing;
using namespace ov::element;

namespace {
struct TestVisitor : public ov::element::NotSupported<bool> {
    using ov::element::NotSupported<bool>::visit;

    template <ov::element::Type_t ET>
    static result_type visit(int x) {
        return true;
    }
};

template <>
TestVisitor::result_type TestVisitor::visit<ov::element::Type_t::i16>(int x) {
    return false;
}

struct TestVisitorNoArgs : public ov::element::NoAction<int, -1> {
    using ov::element::NoAction<int, -1>::visit;

    template <ov::element::Type_t ET>
    static result_type visit() {
        return 10;
    }
};

struct TestVisitorVoidReturn : public ov::element::NoAction<void> {
    using ov::element::NoAction<void>::visit;

    template <ov::element::Type_t ET>
    static result_type visit(int x, int y) {
        test_value = x + y;
    }

    static int test_value;
};

int TestVisitorVoidReturn::test_value;
}  // namespace

class IfTypeOfTest : public Test {
protected:
    void SetUp() override {
        TestVisitorVoidReturn::test_value = 0;
    }
};

TEST_F(IfTypeOfTest, throw_if_not_supported_type) {
    OV_EXPECT_THROW((ov::element::IfTypeOf<i64>::apply<TestVisitor>(u8, 10)),
                    ov::Exception,
                    HasSubstr("Element not supported"));

    OV_EXPECT_THROW((ov::element::IfTypeOf<i64, f32, u16, i32, i8>::apply<TestVisitor>(u8, 10)),
                    ov::Exception,
                    HasSubstr("Element not supported"));
}

TEST_F(IfTypeOfTest, action_for_single_supported_type) {
    const auto result = ov::element::IfTypeOf<f32>::apply<TestVisitor>(ov::element::f32, 10);
    EXPECT_TRUE(result);
}

TEST_F(IfTypeOfTest, action_for_if_multiple_supported_types) {
    const auto result = ov::element::IfTypeOf<u32, f32, i16>::apply<TestVisitor>(f32, 2);
    EXPECT_TRUE(result);
}

TEST_F(IfTypeOfTest, special_action_if_single_supported_type) {
    const auto result = ov::element::IfTypeOf<i16>::apply<TestVisitor>(i16, 20);
    EXPECT_FALSE(result);
}

TEST_F(IfTypeOfTest, special_action_if_multiple_supported_types) {
    const auto result = ov::element::IfTypeOf<u32, i16, f32>::apply<TestVisitor>(i16, 10);
    EXPECT_FALSE(result);
}

TEST_F(IfTypeOfTest, default_action_for_unsupported_type) {
    const auto result = ov::element::IfTypeOf<u32, f32, i16>::apply<TestVisitorNoArgs>(f16);
    EXPECT_EQ(result, -1);
}

TEST_F(IfTypeOfTest, apply_action_for_visitor_with_no_args) {
    const auto result = ov::element::IfTypeOf<f32, u32, i16>::apply<TestVisitorNoArgs>(u32);
    EXPECT_EQ(result, 10);
}

TEST_F(IfTypeOfTest, apply_action_for_void_return_visitor) {
    ov::element::IfTypeOf<f32, u32, i16>::apply<TestVisitorVoidReturn>(u32, 2, 7);
    EXPECT_EQ(TestVisitorVoidReturn::test_value, 9);
}

TEST_F(IfTypeOfTest, apply_default_action_for_void_return_visitor) {
    ov::element::IfTypeOf<f32, u32>::apply<TestVisitorVoidReturn>(i32, 2, 7);
    EXPECT_EQ(TestVisitorVoidReturn::test_value, 0);
}
