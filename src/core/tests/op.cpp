// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset.hpp"

using namespace std;
using namespace ov;

TEST(op, is_op) {
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    EXPECT_TRUE(op::util::is_parameter(arg0));
}

TEST(op, is_parameter) {
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    auto t0 = make_shared<op::v1::Add>(arg0, arg0);
    ASSERT_NE(nullptr, t0);
    EXPECT_FALSE(op::util::is_parameter(t0));
}

TEST(op, opset_multi_thread) {
    auto doTest = [&](std::function<const ov::OpSet&()> fun) {
        std::atomic<const ov::OpSet*> opset{nullptr};
        std::atomic_bool failed{false};
        auto threadFun = [&]() {
            const ov::OpSet* op = &fun();
            const ov::OpSet* current = opset;
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
    doTest(ov::get_opset1);
    doTest(ov::get_opset2);
    doTest(ov::get_opset3);
    doTest(ov::get_opset4);
    doTest(ov::get_opset5);
    doTest(ov::get_opset6);
    doTest(ov::get_opset7);
    doTest(ov::get_opset8);
    doTest(ov::get_opset9);
    doTest(ov::get_opset10);
    doTest(ov::get_opset11);
    doTest(ov::get_opset12);
    doTest(ov::get_opset13);
    doTest(ov::get_opset14);
    doTest(ov::get_opset15);
}
