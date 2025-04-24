// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/opsets/opset.hpp"

#ifdef SELECTIVE_BUILD_ANALYZER
#    define SELECTIVE_BUILD_ANALYZER_ON
#    undef SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD)
#    define SELECTIVE_BUILD_ON
#    undef SELECTIVE_BUILD
#endif

#define SELECTIVE_BUILD

#include "element_visitor.hpp"
#include "itt.hpp"

using namespace std;

TEST(conditional_compilation, disabled_op_scope) {
#define ov_op_Scope0 1
    int n = 0;

    // Simple Scope0 is enabled
    OV_OP_SCOPE(Scope0);
    n = 42;
    EXPECT_EQ(n, 42);

    // Simple Scope1 is disabled and throws exception
    ASSERT_THROW(OV_OP_SCOPE(Scope1), ov::Exception);
#undef ov_op_Scope0
}

TEST(conditional_compilation, disabled_Constant_in_opset) {
#define ov_opset_test_opset3_Abs 1
    ov::OpSet opset("test_opset3");
    INSERT_OP(test_opset3, Abs, ov::op::v0);
    EXPECT_NE(opset.create("Abs"), nullptr);
    EXPECT_NE(opset.create_insensitive("Abs"), nullptr);

    INSERT_OP(test_opset3, Constant, ov::op::v0);
    EXPECT_EQ(opset.create("Constant"), nullptr);
    EXPECT_EQ(opset.create_insensitive("Constant"), nullptr);
#undef ov_opset_test_opset3_Abs
}

namespace {
struct TestVisitor : public ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <ov::element::Type_t ET>
    static result_type visit(int x) {
        return true;
    }
};
}  // namespace

TEST(conditional_compilation, IF_TYPE_OF_element_type_on_cc_list) {
#define TYPE_LIST_ov_eval_enabled_test_1 1
#define TYPE_LIST_ov_eval_test_1         ::ov::element::f32, ::ov::element::u64

    using namespace ov::element;
    const auto result = IF_TYPE_OF(test_1, OV_PP_ET_LIST(f32), TestVisitor, ov::element::f32, 10);
    EXPECT_TRUE(result);

#undef TYPE_LIST_ov_eval_enabled_test_1
#undef TYPE_LIST_ov_eval_test_1
}

TEST(conditional_compilation, IF_TYPE_OF_element_type_not_on_cc_list) {
#define TYPE_LIST_ov_eval_enabled_test_1 1
#define TYPE_LIST_ov_eval_test_1         f16

    using namespace ov::element;
    const auto result = IF_TYPE_OF(test_1, OV_PP_ET_LIST(f32), TestVisitor, ov::element::f32, 10);
    EXPECT_FALSE(result);

#undef TYPE_LIST_ov_eval_enabled_test_1
#undef TYPE_LIST_ov_eval_test_1
}

TEST(conditional_compilation, IF_TYPE_OF_no_element_list) {
    const auto result = IF_TYPE_OF(test_1, OV_PP_ET_LIST(f32), TestVisitor, ov::element::f32, 10);
    EXPECT_FALSE(result);
}
#undef SELECTIVE_BUILD

#ifdef SELECTIVE_BUILD_ANALYZER_ON
#    define SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD_ON)
#    define SELECTIVE_BUILD
#endif
