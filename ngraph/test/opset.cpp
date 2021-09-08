// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset.hpp"

#include <gtest/gtest.h>

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"

TEST(opset, opset1) {
    auto op = std::make_shared<ov::opset1::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset1_dump) {
    const auto& opset = ov::get_opset1();
    std::cout << "All opset1 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(110, opset.get_types_info().size());
}

TEST(opset, opset2) {
    auto op = std::make_shared<ov::opset2::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset2_dump) {
    const auto& opset = ov::get_opset2();
    std::cout << "All opset2 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(112, opset.get_types_info().size());
}

TEST(opset, opset3) {
    auto op = std::make_shared<ov::opset3::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset3_dump) {
    const auto& opset = ov::get_opset3();
    std::cout << "All opset3 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(127, opset.get_types_info().size());
}

TEST(opset, opset4) {
    auto op = std::make_shared<ov::opset4::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset4_dump) {
    const auto& opset = ov::get_opset4();
    std::cout << "All opset4 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(137, opset.get_types_info().size());
}

TEST(opset, opset5) {
    auto op = std::make_shared<ov::opset5::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset5_dump) {
    const auto& opset = ov::get_opset5();
    std::cout << "All opset5 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(145, opset.get_types_info().size());
}

TEST(opset, opset6) {
    auto op = std::make_shared<ov::opset6::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset6_dump) {
    const auto& opset = ov::get_opset6();
    std::cout << "All opset6 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(152, opset.get_types_info().size());
}

TEST(opset, opset7) {
    auto op = std::make_shared<ov::opset7::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset7_dump) {
    const auto& opset = ov::get_opset7();
    std::cout << "All opset7 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(156, opset.get_types_info().size());
}

TEST(opset, opset8) {
    auto op = std::make_shared<ov::opset8::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset8_dump) {
    const auto& opset = ov::get_opset8();
    std::cout << "All opset8 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(162, opset.get_types_info().size());
}
