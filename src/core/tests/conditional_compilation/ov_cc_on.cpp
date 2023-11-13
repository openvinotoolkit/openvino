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

#define SELECTIVE_BUILD

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

#undef SELECTIVE_BUILD

#ifdef SELECTIVE_BUILD_ANALYZER_ON
#    define SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD_ON)
#    define SELECTIVE_BUILD
#endif
