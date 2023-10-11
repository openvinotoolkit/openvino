// Copyright (C) 2018-2023 Intel Corporation
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

#include "itt.hpp"

using namespace std;

TEST(conditional_compilation, op_scope_with_disabled_cc) {
    int n = 0;

    // Simple Scope0 is enabled
    OV_OP_SCOPE(Scope0);
    n = 42;
    EXPECT_EQ(n, 42);

    // Simple Scope1 is enabled regardless of macros
    OV_OP_SCOPE(Scope1);
    n = 43;
    EXPECT_EQ(n, 43);
}

TEST(conditional_compilation, all_ops_enabled_in_opset) {
    ov::OpSet opset("test_opset2");
    INSERT_OP(test_opset2, Abs, ov::op::v0);
    EXPECT_NE(opset.create("Abs"), nullptr);
    EXPECT_NE(opset.create_insensitive("Abs"), nullptr);

    INSERT_OP(test_opset2, Constant, ov::op::v0);
    EXPECT_NE(opset.create("Constant"), nullptr);
    EXPECT_NE(opset.create_insensitive("Constant"), nullptr);
}

#ifdef SELECTIVE_BUILD_ANALYZER_ON
#    define SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD_ON)
#    define SELECTIVE_BUILD
#endif
