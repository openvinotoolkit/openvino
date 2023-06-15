// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "element_visitor.hpp"

using namespace testing;
using namespace ov::element;

struct TestVisitor : public ov::element::NotSupported<bool> {
    using ov::element::NotSupported<bool>::operator();

    template <ov::element::Type_t ET>
    result_type operator()(int x) {
        return true;
    }
};

template <>
TestVisitor::result_type TestVisitor::operator()<ov::element::Type_t::i16>(int x) {
    return false;
}

struct TestVisitorNoArgs : public ov::element::NoAction<int, -1> {
    using ov::element::NoAction<int, -1>::operator();

    template <ov::element::Type_t ET>
    result_type operator()() {
        return 10;
    }
};

struct TestVisitorVoidReturn : public ov::element::NoAction<void> {
    using ov::element::NoAction<void>::operator();

    template <ov::element::Type_t ET>
    result_type operator()(int x, int y) {
        internal_value = x + y;
    }

    int internal_value = 0;
};

class VisitorForSupportedElement : public Test {
protected:
    TestVisitor visitor;
    TestVisitorNoArgs no_args_visitor;
    TestVisitorVoidReturn void_ret_visitor;
};

TEST_F(VisitorForSupportedElement, throw_if_not_supported_type) {
    OV_EXPECT_THROW((ov::element::Supported<i64>::apply(u8, visitor, 10)),
                    ov::Exception,
                    HasSubstr("Element not supported"));

    OV_EXPECT_THROW((ov::element::Supported<i64, f32, u16, i32, i8>::apply(u8, visitor, 10)),
                    ov::Exception,
                    HasSubstr("Element not supported"));
}

TEST_F(VisitorForSupportedElement, action_for_single_supported_type) {
    const auto result = ov::element::Supported<f32>::apply(ov::element::f32, visitor, 10);
    EXPECT_TRUE(result);
}

TEST_F(VisitorForSupportedElement, action_for_if_multiple_supported_types) {
    const auto result = ov::element::Supported<u32, f32, i16>::apply(f32, visitor, 2);
    EXPECT_TRUE(result);
}

TEST_F(VisitorForSupportedElement, special_action_if_single_supported_type) {
    const auto result = ov::element::Supported<i16>::apply(i16, visitor, 20);
    EXPECT_FALSE(result);
}

TEST_F(VisitorForSupportedElement, special_action_if_multiple_supported_types) {
    const auto result = ov::element::Supported<u32, i16, f32>::apply(i16, visitor, 10);
    EXPECT_FALSE(result);
}

TEST_F(VisitorForSupportedElement, default_action_for_unsupported_type) {
    const auto result = ov::element::Supported<u32, f32, i16>::apply(f16, no_args_visitor);
    EXPECT_EQ(result, -1);
}

TEST_F(VisitorForSupportedElement, apply_action_for_visitor_with_no_args) {
    const auto result = ov::element::Supported<f32, u32, i16>::apply(u32, no_args_visitor);
    EXPECT_EQ(result, 10);
}

TEST_F(VisitorForSupportedElement, apply_action_for_void_return_visitor) {
    ov::element::Supported<f32, u32, i16>::apply(u32, void_ret_visitor, 2, 7);
    EXPECT_EQ(void_ret_visitor.internal_value, 9);
}

TEST_F(VisitorForSupportedElement, apply_default_action_for_void_return_visitor) {
    ov::element::Supported<f32, u32>::apply(i32, void_ret_visitor, 2, 7);
    EXPECT_EQ(void_ret_visitor.internal_value, 0);
}
