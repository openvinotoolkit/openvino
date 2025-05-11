// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mod.hpp"

#include "arithmetic_ops.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"

using Type = ::testing::Types<ov::op::v1::Mod>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_mod, ArithmeticOperator, Type);

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::op::v0::Squeeze;
using ov::op::v3::Broadcast;
using ov::op::v3::ShapeOf;

class TypePropModV1Test : public TypePropOpTest<op::v1::Mod> {};

TEST_F(TypePropModV1Test, preserve_constant_data_on_inputs) {
    const auto a = Constant::create(ov::element::i32, ov::Shape{4}, {4, 10, 22, 5});
    const auto b = Constant::create(ov::element::i32, ov::Shape{4}, {3, 4, 8, 3});
    const auto op = make_op(a, b);

    const auto param = std::make_shared<Parameter>(ov::element::i32, ov::Shape{1});
    auto bc = std::make_shared<Broadcast>(param, op, ov::op::BroadcastType::BIDIRECTIONAL);
    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, ov::PartialShape({1, 2, 6, 2}));
}

TEST_F(TypePropModV1Test, preserve_partial_values_on_inputs) {
    const auto a = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{{5, 6}, 22, {3, 7}, -1, {7, 9}});
    const auto b = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{3, {12, 18}, {4, 6}, -1, {0, 4}});
    const auto op = make_op(std::make_shared<ShapeOf>(a), std::make_shared<ShapeOf>(b));

    const auto param = std::make_shared<Parameter>(ov::element::i64, ov::Shape{1});
    auto bc = std::make_shared<Broadcast>(param, op, ov::op::BroadcastType::BIDIRECTIONAL);

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, ov::PartialShape({{0, 2}, {4, 10}, {0, 5}, -1, -1}));
}

TEST_F(TypePropModV1Test, preserve_partial_values_when_m_is_interval_scalar) {
    const auto a = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{{5, 6}, 22, {3, 7}, -1, {7, 9}});
    const auto b = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{{12, 18}});
    const auto b_scalar = std::make_shared<Squeeze>(std::make_shared<ShapeOf>(b));
    const auto op = make_op(std::make_shared<ShapeOf>(a), b_scalar);

    const auto param = std::make_shared<Parameter>(ov::element::i64, ov::Shape{1});
    auto bc = std::make_shared<Broadcast>(param, op, ov::op::BroadcastType::BIDIRECTIONAL);

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, ov::PartialShape({{5, 6}, {4, 10}, {3, 7}, -1, {7, 9}}));
}

TEST_F(TypePropModV1Test, preserve_partial_values_when_value_is_interval_scalar) {
    const auto a = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{{3, 7}});
    const auto b = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{3, {12, 18}, {4, 6}, -1, {0, 4}});
    const auto a_scalar = std::make_shared<Squeeze>(std::make_shared<ShapeOf>(a));
    const auto op = make_op(a_scalar, std::make_shared<ShapeOf>(b));

    const auto param = std::make_shared<Parameter>(ov::element::i64, ov::Shape{1});
    auto bc = std::make_shared<Broadcast>(param, op, ov::op::BroadcastType::BIDIRECTIONAL);

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, ov::PartialShape({{0, 2}, {3, 7}, {0, 5}, -1, -1}));
}

// test params as {a, b, exp_result}
using IntervalModuloParams = std::tuple<ov::Dimension, ov::Dimension, ov::Dimension>;

class SingleDimModV1Test : public TypePropModV1Test, public testing::WithParamInterface<IntervalModuloParams> {
protected:
    void SetUp() override {
        std::tie(a_dim, b_dim, exp_dim) = GetParam();
    }

    ov::Dimension a_dim, b_dim, exp_dim;
};

const auto v_and_m_static = testing::Values(IntervalModuloParams{{0, 0}, {1, 1}, {0, 0}},
                                            IntervalModuloParams{{0, 0}, {9, 9}, {0, 0}},
                                            IntervalModuloParams{{0, 0}, {1000, 1000}, {0, 0}},
                                            IntervalModuloParams{{10, 10}, {3, 3}, {1, 1}},
                                            IntervalModuloParams{{10, 10}, {6, 6}, {4, 4}},
                                            IntervalModuloParams{{10, 10}, {5, 5}, {0, 0}},
                                            IntervalModuloParams{{10, 10}, {15, 15}, {10, 10}});

const auto v_interval_m_static = testing::Values(IntervalModuloParams{{6, 7}, {4, 4}, {2, 3}},
                                                 IntervalModuloParams{{6, 8}, {4, 4}, {0, 3}},  // Result [0,2,3]
                                                 IntervalModuloParams{{6, 8}, {10, 10}, {6, 8}},
                                                 IntervalModuloParams{{6, 8}, {7, 7}, {0, 6}},
                                                 IntervalModuloParams{{4, 8}, {7, 7}, {0, 6}},
                                                 IntervalModuloParams{{15, 16}, {7, 7}, {1, 2}},
                                                 IntervalModuloParams{{5, 20}, {5, 5}, {0, 4}},

                                                 IntervalModuloParams{{5, 10}, {7, 7}, {0, 6}});

const auto v_static_m_interval = testing::Values(IntervalModuloParams{{0, 0}, {3, 13}, {0, 0}},
                                                 IntervalModuloParams{{10, 10}, {2, 4}, {0, 3}},
                                                 IntervalModuloParams{{10, 10}, {2, 6}, {0, 4}},
                                                 IntervalModuloParams{{10, 10}, {6, 9}, {1, 4}},
                                                 IntervalModuloParams{{10, 10}, {9, 11}, {0, 10}},
                                                 IntervalModuloParams{{10, 10}, {3, 11}, {0, 10}},
                                                 IntervalModuloParams{{10, 10}, {3, 10}, {0, 9}},
                                                 IntervalModuloParams{{10, 10}, {7, 8}, {2, 3}},
                                                 IntervalModuloParams{{100, 100}, {2, 20}, {0, 19}},
                                                 // can be estimated accurate as only two results are possible
                                                 IntervalModuloParams{{100, 100}, {15, 16}, {4, 10}},
                                                 // can not be estimated accurate as there are three results [10,4,15]
                                                 // Requires to calculate all possibilities and pick min, max
                                                 IntervalModuloParams{{100, 100}, {15, 17}, {0, 16}});

const auto v_and_m_intervals = testing::Values(IntervalModuloParams{{1, 10}, {2, 9}, {0, 8}},
                                               IntervalModuloParams{{1, 10}, {6, 9}, {0, 8}},
                                               IntervalModuloParams{{1, 10}, {2, 12}, {0, 10}},
                                               IntervalModuloParams{{1, 10}, {6, 12}, {0, 10}},
                                               IntervalModuloParams{{1, 10}, {11, 12}, {1, 10}},
                                               IntervalModuloParams{{1, 10}, {11, 15}, {1, 10}},
                                               IntervalModuloParams{{4, 10}, {10, 13}, {0, 10}},
                                               IntervalModuloParams{{10, 20}, {3, 5}, {0, 4}},
                                               IntervalModuloParams{{10, 10}, {3, 10}, {0, 9}},
                                               IntervalModuloParams{{5, 20}, {5, 10}, {0, 9}},
                                               IntervalModuloParams{{10, 100}, {3, 20}, {0, 19}},
                                               IntervalModuloParams{{10, 100}, {2, 20}, {0, 19}},
                                               IntervalModuloParams{{10, 100}, {51, 60}, {0, 59}});

// If input is infinite or m has 0 then output is undefined.
const auto v_and_m_special_values = testing::Values(IntervalModuloParams{{0, -1}, {5, 5}, {0, -1}},
                                                    IntervalModuloParams{{10, -1}, {4, 4}, {0, -1}},
                                                    // Evaluate low/up return [0, max]
                                                    // but evaluate both bounds return [0] as `m` has same bounds
                                                    IntervalModuloParams{{11, 11}, {0, 0}, {0, 0}},
                                                    IntervalModuloParams{{11, 11}, {0, 5}, {0, -1}},
                                                    IntervalModuloParams{{11, 20}, {0, 5}, {0, -1}},
                                                    IntervalModuloParams{{11, 20}, {0, -1}, {0, -1}},
                                                    IntervalModuloParams{{0, -1}, {0, -1}, {0, -1}});

INSTANTIATE_TEST_SUITE_P(v_and_m_static, SingleDimModV1Test, v_and_m_static);
INSTANTIATE_TEST_SUITE_P(value_interval_m_static, SingleDimModV1Test, v_interval_m_static);
INSTANTIATE_TEST_SUITE_P(value_static_m_interval, SingleDimModV1Test, v_static_m_interval);
INSTANTIATE_TEST_SUITE_P(value_and_m_as_intervals, SingleDimModV1Test, v_and_m_intervals);
INSTANTIATE_TEST_SUITE_P(value_and_m_special_values, SingleDimModV1Test, v_and_m_special_values);

TEST_P(SingleDimModV1Test, preserve_value_on_inputs_i64) {
    constexpr auto et = ov::element::i64;
    const auto a = std::make_shared<Parameter>(et, ov::PartialShape{a_dim});
    const auto b = std::make_shared<Parameter>(et, ov::PartialShape{b_dim});
    const auto op = make_op(std::make_shared<ShapeOf>(a), std::make_shared<ShapeOf>(b));

    const auto param = std::make_shared<Parameter>(et, ov::Shape{1});
    const auto bc = std::make_shared<Broadcast>(param, op, ov::op::BroadcastType::BIDIRECTIONAL);
    const auto& output_shape = bc->get_output_partial_shape(0);

    EXPECT_EQ(output_shape, ov::PartialShape({exp_dim}));
}

TEST_P(SingleDimModV1Test, preserve_value_on_inputs_i32) {
    constexpr auto et = ov::element::i32;
    const auto a = std::make_shared<Parameter>(et, ov::PartialShape{a_dim});
    const auto b = std::make_shared<Parameter>(et, ov::PartialShape{b_dim});
    const auto op = make_op(std::make_shared<ShapeOf>(a, et), std::make_shared<ShapeOf>(b, et));

    const auto param = std::make_shared<Parameter>(et, ov::Shape{1});
    const auto bc = std::make_shared<Broadcast>(param, op, ov::op::BroadcastType::BIDIRECTIONAL);
    const auto& output_shape = bc->get_output_partial_shape(0);

    EXPECT_EQ(output_shape, ov::PartialShape({exp_dim}));
}
