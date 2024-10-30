// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset.hpp"

#include <gtest/gtest.h>

#include "openvino/op/op.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset14.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"

struct OpsetTestParams {
    using OpsetGetterFunction = std::function<const ov::OpSet&()>;
    OpsetTestParams(const OpsetGetterFunction& opset_getter_, const uint32_t expected_ops_count_)
        : opset_getter{opset_getter_},
          expected_ops_count{expected_ops_count_} {}

    OpsetGetterFunction opset_getter;
    uint32_t expected_ops_count;
};

class OpsetTests : public testing::TestWithParam<OpsetTestParams> {};

struct OpsetTestNameGenerator {
    std::string operator()(const testing::TestParamInfo<OpsetTestParams>& info) const {
        return "opset" + std::to_string(info.index + 1);
    }
};

TEST_P(OpsetTests, create_parameter) {
    const auto& params = GetParam();
    const auto op = std::unique_ptr<ov::Node>(params.opset_getter().create("Parameter"));
    ASSERT_NE(nullptr, op);
    EXPECT_TRUE(ov::op::util::is_parameter(op.get()));
}

TEST_P(OpsetTests, opset_dump) {
    const auto& params = GetParam();
    const auto& opset = params.opset_getter();
    std::cout << "All opset operations: ";
    for (const auto& t : opset.get_types_info()) {
        std::cout << t.name << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(params.expected_ops_count, opset.get_types_info().size());
}

INSTANTIATE_TEST_SUITE_P(opset,
                         OpsetTests,
                         testing::Values(OpsetTestParams{ov::get_opset1, 110},
                                         OpsetTestParams{ov::get_opset2, 112},
                                         OpsetTestParams{ov::get_opset3, 127},
                                         OpsetTestParams{ov::get_opset4, 137},
                                         OpsetTestParams{ov::get_opset5, 145},
                                         OpsetTestParams{ov::get_opset6, 152},
                                         OpsetTestParams{ov::get_opset7, 156},
                                         OpsetTestParams{ov::get_opset8, 167},
                                         OpsetTestParams{ov::get_opset9, 173},
                                         OpsetTestParams{ov::get_opset10, 177},
                                         OpsetTestParams{ov::get_opset11, 177},
                                         OpsetTestParams{ov::get_opset12, 178},
                                         OpsetTestParams{ov::get_opset13, 186},
                                         OpsetTestParams{ov::get_opset14, 188},
                                         OpsetTestParams{ov::get_opset15, 199}),
                         OpsetTestNameGenerator{});

class MyOpOld : public ov::op::Op {
public:
    static constexpr ov::DiscreteTypeInfo type_info{"MyOpOld"};
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
    MyOpNewFromOld() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return nullptr;
    }
};

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
    opset.insert<MyOpIncorrect>();
    opset.insert<MyOpNewFromOld>();
    opset.insert<MyOpNew>();
    EXPECT_EQ(opset.get_types_info().size(), 3);
    EXPECT_TRUE(opset.contains_type(std::string("MyOpNewFromOld")));
    EXPECT_TRUE(opset.contains_type(std::string("MyOpNew")));
    EXPECT_TRUE(opset.contains_type(std::string("MyOpIncorrect")));
}
