// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset.hpp"

#include <gtest/gtest.h>

#include "openvino/op/op.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"

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
    ASSERT_EQ(167, opset.get_types_info().size());
}

TEST(opset, opset9) {
    auto op = std::make_shared<ov::opset9::Parameter>();
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op));
}

TEST(opset, opset9_dump) {
    const auto& opset = ov::get_opset9();
    std::cout << "All opset9 operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(172, opset.get_types_info().size());
}

class MyOpOld : public ov::op::Op {
public:
    static constexpr ov::DiscreteTypeInfo type_info{"MyOpOld", static_cast<uint64_t>(0)};
    const ov::DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
    MyOpOld() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return nullptr;
    }
};

constexpr ov::DiscreteTypeInfo MyOpOld::type_info;

class MyOpNewFromOld : public MyOpOld {
public:
    OPENVINO_OP("MyOpNewFromOld", "custom_opset", MyOpOld);
    BWDCMP_RTTI_DECLARATION;
    MyOpNewFromOld() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return nullptr;
    }
};

BWDCMP_RTTI_DEFINITION(MyOpNewFromOld);

class MyOpIncorrect : public MyOpOld {
public:
    OPENVINO_OP("MyOpIncorrect", "custom_opset", MyOpOld);
    MyOpIncorrect() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return nullptr;
    }
};

class MyOpNew : public ov::op::Op {
public:
    OPENVINO_OP("MyOpNew", "custom_opset", MyOpOld);
    MyOpNew() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return nullptr;
    }
};

TEST(opset, custom_opset) {
    ov::OpSet opset;
#ifndef OPENVINO_STATIC_LIBRARY
    opset.insert<MyOpOld>();
#endif
    opset.insert<MyOpIncorrect>();
    opset.insert<MyOpNewFromOld>();
    opset.insert<MyOpNew>();
#ifdef OPENVINO_STATIC_LIBRARY
    EXPECT_EQ(opset.get_types_info().size(), 2);
#else
    EXPECT_EQ(opset.get_types_info().size(), 3);
    EXPECT_TRUE(opset.contains_type("MyOpOld"));
    // TODO: why is it not registered?
    EXPECT_TRUE(opset.contains_type("MyOpNewFromOld"));
#endif
    EXPECT_TRUE(opset.contains_type("MyOpNew"));
    EXPECT_FALSE(opset.contains_type("MyOpIncorrect"));
}
