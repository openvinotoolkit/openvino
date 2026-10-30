// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

namespace ov::test {
using op::v0::Constant, op::v0::Parameter;

class TypePropQSAIndexerTest : public TypePropOpTest<op::internal::QSAIndexer> {
protected:
    using Config = op::internal::QSAIndexer::Config;

    static std::shared_ptr<Parameter> param(element::Type et, const PartialShape& shape) {
        return std::make_shared<Parameter>(et, shape);
    }

    static std::shared_ptr<Constant> disabled() {
        return Constant::create(element::f32, Shape{0}, std::vector<float>{});
    }

    /// Inputs with query [batch, heads, sequence, head_size] and a rotary dimension of 16.
    static OutputVector make_inputs(const PartialShape& query_shape = PartialShape{2, 4, 3, 128},
                                    const Dimension& head_size = 128) {
        return {param(element::f32, query_shape),
                param(element::f32, PartialShape{2, 3, head_size}),
                param(element::f32, PartialShape{2, 3, 16}),
                param(element::f32, PartialShape{2, 3, 16}),
                param(element::i32, PartialShape{2}),
                param(element::f32, PartialShape{head_size}),
                param(element::f32, PartialShape{head_size}),
                param(element::f32, PartialShape{2, 4, head_size}),
                param(element::f32, PartialShape{2, 3, head_size}),
                param(element::f32, PartialShape{2, 32})};
    }

    static Config config() {
        return Config{4, 16, 1, 1e-6f, 1e-6f, true};
    }
};

TEST_F(TypePropQSAIndexerTest, infers_selection_and_state_shapes) {
    const auto op = make_op(make_inputs(), config());
    EXPECT_EQ(op->get_output_size(), 5);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 3, 4}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 1, 3}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{2, 4, 128}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{2, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{2, 32}));
}

TEST_F(TypePropQSAIndexerTest, dynamic_batch_and_sequence) {
    auto inputs = make_inputs(PartialShape{-1, 4, -1, 128});
    inputs[1] = param(element::f32, PartialShape{-1, -1, 128});
    inputs[2] = param(element::f32, PartialShape{-1, -1, 16});
    inputs[3] = param(element::f32, PartialShape{-1, -1, 16});
    inputs[4] = param(element::i32, PartialShape{1});
    inputs[7] = param(element::f32, PartialShape{-1, -1, 128});
    inputs[8] = param(element::f32, PartialShape{-1, 3, 128});
    inputs[9] = param(element::f32, PartialShape{-1, 32});
    const auto op = make_op(inputs, config());
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 1, -1, 4}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{-1, -1, 128}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{-1, 3, 128}));
}

TEST_F(TypePropQSAIndexerTest, rope_disabled_by_empty_inputs) {
    auto inputs = make_inputs();
    inputs[2] = disabled();
    inputs[3] = disabled();
    inputs[9] = disabled();
    const auto op = make_op(inputs, config());
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{2, 0}));
}

TEST_F(TypePropQSAIndexerTest, head_count_and_head_size_derived_from_inputs) {
    const auto op = make_op(make_inputs(PartialShape{2, 8, 3, 64}, 64), config());
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 3, 4}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{2, 4, 64}));
}

TEST_F(TypePropQSAIndexerTest, rotary_dimension_larger_than_head_size) {
    OV_EXPECT_THROW(std::ignore = make_op(make_inputs(PartialShape{2, 4, 3, 8}, 8), config()),
                    NodeValidationFailure,
                    testing::HasSubstr("Rotary dimension must not exceed the head size"));
}

TEST_F(TypePropQSAIndexerTest, non_causal_configuration) {
    auto non_causal = config();
    non_causal.is_causal = false;
    EXPECT_THROW(std::ignore = make_op(make_inputs(), non_causal), NodeValidationFailure);
}

TEST_F(TypePropQSAIndexerTest, incorrect_number_of_inputs) {
    auto inputs = make_inputs();
    inputs.pop_back();
    EXPECT_THROW(std::ignore = make_op(inputs, config()), NodeValidationFailure);
}

TEST_F(TypePropQSAIndexerTest, incorrect_epsilon) {
    auto zero_epsilon = config();
    zero_epsilon.q_epsilon = 0.0f;
    EXPECT_THROW(std::ignore = make_op(make_inputs(), zero_epsilon), NodeValidationFailure);
}
}  // namespace ov::test
