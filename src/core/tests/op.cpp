// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/variant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/sin.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(op, is_op) {
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    EXPECT_TRUE(op::is_parameter(arg0));
}

TEST(op, is_parameter) {
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    auto t0 = make_shared<op::v1::Add>(arg0, arg0);
    ASSERT_NE(nullptr, t0);
    EXPECT_FALSE(op::is_parameter(t0));
}

TEST(op, opset_multi_thread) {
    auto doTest = [&](std::function<const ngraph::OpSet&()> fun) {
        std::atomic<const ngraph::OpSet*> opset{nullptr};
        std::atomic_bool failed{false};
        auto threadFun = [&]() {
            const ngraph::OpSet* op = &fun();
            const ngraph::OpSet* current = opset;
            do {
                if (current != nullptr && current != op) {
                    failed = true;
                    break;
                }
            } while (opset.compare_exchange_strong(op, current));
        };
        std::thread t1{threadFun};
        std::thread t2{threadFun};
        t1.join();
        t2.join();
        ASSERT_FALSE(failed);
    };
    doTest(ngraph::get_opset1);
    doTest(ngraph::get_opset2);
    doTest(ngraph::get_opset3);
    doTest(ngraph::get_opset4);
    doTest(ngraph::get_opset5);
    doTest(ngraph::get_opset6);
    doTest(ngraph::get_opset7);
}

TEST(op, cos_op_use_different_opsets) {
    ov::op::opset1::Cos op1;
    ov::op::opset2::Cos op2;
    ov::op::opset3::Cos op3;
    ov::op::opset4::Cos op4;
    ov::op::opset5::Cos op5;
    ov::op::opset6::Cos op6;
    ov::op::opset7::Cos op7;
    ov::op::opset8::Cos op8;
}

TEST(op, sin_op_use_different_opsets) {
    ov::op::opset1::Sin op1;
    ov::op::opset2::Sin op2;
    ov::op::opset3::Sin op3;
    ov::op::opset4::Sin op4;
    ov::op::opset5::Sin op5;
    ov::op::opset6::Sin op6;
    ov::op::opset7::Sin op7;
    ov::op::opset8::Sin op8;
}
