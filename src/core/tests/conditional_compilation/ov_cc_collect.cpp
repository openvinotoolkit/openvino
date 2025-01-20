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

#define SELECTIVE_BUILD_ANALYZER

#include "element_visitor.hpp"
#include "itt.hpp"

using namespace std;

TEST(conditional_compilation, collect_op_scope) {
#define ov_op_Scope0 1
    int n = 0;

    // Simple Scope0 is enabled
    OV_OP_SCOPE(Scope0);
    n = 42;
    EXPECT_EQ(n, 42);

    // Simple Scope1 is enabled regardless of macros
    OV_OP_SCOPE(Scope1);
    n = 43;
    EXPECT_EQ(n, 43);
#undef ov_op_Scope0
}

TEST(conditional_compilation, collect_ops_in_opset) {
#define ov_opset_test_opset1_Abs 1
    ov::OpSet opset("test_opset1");
    INSERT_OP(test_opset1, Abs, ov::op::v0);
    EXPECT_NE(opset.create("Abs"), nullptr);
    EXPECT_NE(opset.create_insensitive("Abs"), nullptr);

    INSERT_OP(test_opset1, Constant, ov::op::v0);
    EXPECT_NE(opset.create("Constant"), nullptr);
    EXPECT_NE(opset.create_insensitive("Constant"), nullptr);
#undef ov_opset_test_opset1_Abs
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

TEST(conditional_compilation, IF_TYPE_OF_collect_action_for_supported_element) {
    using namespace ov::element;
    const auto result = IF_TYPE_OF(test_1, OV_PP_ET_LIST(f32), TestVisitor, ov::element::f32, 10);
    EXPECT_TRUE(result);
}

#undef SELECTIVE_BUILD_ANALYZER

#ifdef SELECTIVE_BUILD_ANALYZER_ON
#    define SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD_ON)
#    define SELECTIVE_BUILD
#endif
