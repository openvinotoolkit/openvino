// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_prepare.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

namespace ov::test {
using op::v0::Constant, op::v0::Parameter;
using testing::HasSubstr;

class TypePropQSAPrepareTest : public TypePropOpTest<op::v17::QSAPrepare> {
protected:
    static std::shared_ptr<Parameter> param(element::Type et, const PartialShape& shape) {
        return std::make_shared<Parameter>(et, shape);
    }

    static std::shared_ptr<Constant> disabled(element::Type et = element::f32) {
        return Constant::create(et, Shape{0}, std::vector<float>{});
    }
};

TEST_F(TypePropQSAPrepareTest, default_ctor) {
    const auto op = make_op();
    op->set_key_block_size(4);
    op->set_arguments(OutputVector{param(element::f32, PartialShape{1, 1, 128}),
                                   param(element::f32, PartialShape{1, 2500, 128}),
                                   Constant::create(element::i64, Shape{1}, {10004}),
                                   param(element::f32, PartialShape{1, 1, 64}),
                                   param(element::f32, PartialShape{1, 1, 64}),
                                   param(element::f32, PartialShape{128}),
                                   param(element::f32, PartialShape{1, 3, 128}),
                                   param(element::f32, PartialShape{1, 128})});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 8);
    EXPECT_EQ(op->get_output_size(), 3);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2501, 128}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 128}));
}

TEST_F(TypePropQSAPrepareTest, static_shapes_with_constant_key_length) {
    // The previous call ended with key_length [32, 15], so it cached 8 and 3 complete blocks.
    const auto op = make_op(param(element::f32, PartialShape{2, 8, 128}),
                            param(element::f32, PartialShape{2, 8, 128}),
                            Constant::create(element::i32, Shape{2}, {40, 23}),
                            param(element::f32, PartialShape{2, 8, 64}),
                            param(element::f32, PartialShape{2, 8, 64}),
                            param(element::f32, PartialShape{128}),
                            param(element::f32, PartialShape{2, 3, 128}),
                            param(element::f32, PartialShape{2, 128}),
                            4);

    EXPECT_EQ(op->get_key_block_size(), 4);
    EXPECT_EQ(op->get_epsilon(), 1e-6f);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_element_type(2), element::f32);
    // The longest sequence has 40 valid tokens, that is 10 complete blocks.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 10, 128}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{2, 128}));
}

TEST_F(TypePropQSAPrepareTest, insufficient_cached_block_keys) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 8, 128}),
                                          param(element::f32, PartialShape{2, 4, 128}),
                                          Constant::create(element::i32, Shape{2}, {40, 23}),
                                          param(element::f32, PartialShape{2, 8, 64}),
                                          param(element::f32, PartialShape{2, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{2, 3, 128}),
                                          param(element::f32, PartialShape{2, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of past_block_index_keys must cover the complete blocks"));
}

TEST_F(TypePropQSAPrepareTest, dynamic_key_length_keeps_block_count_dynamic) {
    const auto op = make_op(param(element::f16, PartialShape{2, 8, 128}),
                            param(element::f16, PartialShape{2, -1, 128}),
                            param(element::i32, PartialShape{2}),
                            param(element::f16, PartialShape{2, 8, 64}),
                            param(element::f16, PartialShape{2, 8, 64}),
                            param(element::f16, PartialShape{128}),
                            param(element::f16, PartialShape{2, 3, 128}),
                            param(element::f16, PartialShape{2, 128}),
                            4);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, -1, 128}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{2, 128}));
}

TEST_F(TypePropQSAPrepareTest, dynamic_key_length_is_bounded_by_the_cached_blocks) {
    const auto op = make_op(param(element::f32, PartialShape{1, 8, 128}),
                            param(element::f32, PartialShape{1, 10, 128}),
                            param(element::i32, PartialShape{1}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            param(element::f32, PartialShape{128}),
                            param(element::f32, PartialShape{1, 3, 128}),
                            param(element::f32, PartialShape{1, 128}),
                            4);

    // At most the 10 cached blocks plus ceil(8 / 4) completed by the new tokens.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, Dimension(0, 12), 128}));
}

TEST_F(TypePropQSAPrepareTest, negative_key_length_value) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 8, 128}),
                                          Constant::create(element::i32, Shape{1}, {-5}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The key_length values must be non-negative."));
}

TEST_F(TypePropQSAPrepareTest, rotation_disabled_by_empty_rope) {
    const auto op = make_op(param(element::f32, PartialShape{1, 8, 128}),
                            param(element::f32, PartialShape{1, 0, 128}),
                            param(element::i32, PartialShape{1}),
                            disabled(),
                            disabled(),
                            param(element::f32, PartialShape{128}),
                            param(element::f32, PartialShape{1, 3, 128}),
                            disabled(),
                            4);

    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 0}));
}

TEST_F(TypePropQSAPrepareTest, normalization_disabled_by_empty_weight) {
    const auto op = make_op(param(element::f32, PartialShape{1, 8, 128}),
                            param(element::f32, PartialShape{1, 0, 128}),
                            param(element::i32, PartialShape{1}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            disabled(),
                            param(element::f32, PartialShape{1, 3, 128}),
                            param(element::f32, PartialShape{1, 128}),
                            4);

    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 3, 128}));
}

TEST_F(TypePropQSAPrepareTest, empty_rope_parameter_does_not_disable_the_rotation) {
    // A [0] Parameter is not an empty Constant, so the rotation stays enabled.
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{0}),
                                          param(element::f32, PartialShape{0}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{0}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The shape of rope_cos and rope_sin must be 3D [N, L, Dr]."));
}

TEST_F(TypePropQSAPrepareTest, normalization_disabled_by_multi_dim_empty_constant) {
    const auto op = make_op(param(element::f32, PartialShape{1, 8, 128}),
                            param(element::f32, PartialShape{1, 0, 128}),
                            param(element::i32, PartialShape{1}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            Constant::create(element::f32, Shape{1, 0}, std::vector<float>{}),
                            param(element::f32, PartialShape{1, 3, 128}),
                            param(element::f32, PartialShape{1, 128}),
                            4);

    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 3, 128}));
}

TEST_F(TypePropQSAPrepareTest, multi_dim_empty_k_norm_weight_parameter_is_rejected) {
    // A [1, 0] Parameter is not an empty Constant, so the normalization stays enabled.
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 0}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The shape of k_norm_weight must be 1D [Di], or an empty constant"));
}

TEST_F(TypePropQSAPrepareTest, possibly_empty_k_norm_weight_must_match_head_size) {
    // An interval containing 0 is not an empty Constant, so the input is enabled.
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{Dimension(0, 64)}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The size of k_norm_weight must be equal to the head size of raw_index_keys."));
}

TEST_F(TypePropQSAPrepareTest, dynamic_rope_must_not_be_mixed_with_a_disabled_state) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, -1}),
                                          param(element::f32, PartialShape{1, 8, -1}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          disabled(),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("must be empty constants together to disable the rotation."));
}

TEST_F(TypePropQSAPrepareTest, rope_batch_may_broadcast) {
    const auto op = make_op(param(element::f32, PartialShape{2, 8, 128}),
                            param(element::f32, PartialShape{2, 0, 128}),
                            param(element::i32, PartialShape{2}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            param(element::f32, PartialShape{128}),
                            param(element::f32, PartialShape{2, 3, 128}),
                            param(element::f32, PartialShape{2, 128}),
                            4);

    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 3, 128}));
}

TEST_F(TypePropQSAPrepareTest, incorrect_rope_batch) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 8, 128}),
                                          param(element::f32, PartialShape{2, 0, 128}),
                                          param(element::i32, PartialShape{2}),
                                          param(element::f32, PartialShape{3, 8, 64}),
                                          param(element::f32, PartialShape{3, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{2, 3, 128}),
                                          param(element::f32, PartialShape{2, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The batch dimension of rope_cos must be 1 or the batch of raw_index_keys."));
}

TEST_F(TypePropQSAPrepareTest, odd_number_of_rotated_dimensions) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 65}),
                                          param(element::f32, PartialShape{1, 8, 65}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 130}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of rotated dimensions must be a positive even integer."));
}

TEST_F(TypePropQSAPrepareTest, zero_rotated_dimensions) {
    // An enabled rotation of zero dimensions would be indistinguishable from a disabled one.
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 0}),
                                          param(element::f32, PartialShape{1, 8, 0}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 0}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of rotated dimensions must be a positive even integer."));
}

TEST_F(TypePropQSAPrepareTest, too_many_rotated_dimensions) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 256}),
                                          param(element::f32, PartialShape{1, 8, 256}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 512}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of rotated dimensions must not exceed the head size"));
}

TEST_F(TypePropQSAPrepareTest, incorrect_key_length_size) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 8, 128}),
                                          param(element::f32, PartialShape{2, 0, 128}),
                                          Constant::create(element::i32, Shape{3}, {8, 8, 400}),
                                          param(element::f32, PartialShape{2, 8, 64}),
                                          param(element::f32, PartialShape{2, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{2, 3, 128}),
                                          param(element::f32, PartialShape{2, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The size of key_length must be 1 or the batch of raw_index_keys."));
}

TEST_F(TypePropQSAPrepareTest, dynamic_rank_raw_index_keys) {
    const auto op = make_op(param(element::f32, PartialShape::dynamic()),
                            param(element::f32, PartialShape{1, 0, 128}),
                            param(element::i32, PartialShape{1}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            param(element::f32, PartialShape{1, 8, 64}),
                            param(element::f32, PartialShape{128}),
                            param(element::f32, PartialShape{1, 3, 128}),
                            param(element::f32, PartialShape{1, 128}),
                            4);

    // The batch and the head size are still recovered from the other inputs.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, -1, 128}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 128}));
}

TEST_F(TypePropQSAPrepareTest, symbols_propagation) {
    auto raw_index_keys_shape = PartialShape{-1, 8, -1};
    const auto symbols = set_shape_symbols(raw_index_keys_shape);

    const auto op = make_op(param(element::f32, raw_index_keys_shape),
                            param(element::f32, PartialShape{-1, -1, -1}),
                            param(element::i32, PartialShape{-1}),
                            param(element::f32, PartialShape{-1, 8, 64}),
                            param(element::f32, PartialShape{-1, 8, 64}),
                            param(element::f32, PartialShape{-1}),
                            param(element::f32, PartialShape{-1, 3, -1}),
                            param(element::f32, PartialShape{-1, 128}),
                            4);

    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)),
                testing::ElementsAre(symbols[0], nullptr, symbols[2]));
}

TEST_F(TypePropQSAPrepareTest, incorrect_raw_index_keys_rank) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The shape of raw_index_keys must be 3D [N, L, Di]."));
}

TEST_F(TypePropQSAPrepareTest, head_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 64}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The head size of raw_index_keys and past_block_index_keys must be the same."));
}

TEST_F(TypePropQSAPrepareTest, incorrect_past_incomplete_raw_keys_size) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 4, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of past_incomplete_raw_keys must be key_block_size - 1"));
}

TEST_F(TypePropQSAPrepareTest, incorrect_past_incomplete_rope_size) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 64}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The second dimension of past_incomplete_rope must be 2 * Dr."));
}

TEST_F(TypePropQSAPrepareTest, rope_cos_and_rope_sin_must_be_empty_together) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          disabled(),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          disabled(),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("must be empty constants together to disable the rotation."));
}

TEST_F(TypePropQSAPrepareTest, past_incomplete_rope_must_be_empty_with_the_rope_inputs) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          disabled(),
                                          disabled(),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("must be empty constants together to disable the rotation."));
}

TEST_F(TypePropQSAPrepareTest, past_incomplete_rope_must_not_be_empty_with_the_rotation) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          disabled(),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("must be empty constants together to disable the rotation."));
}

TEST_F(TypePropQSAPrepareTest, incorrect_data_element_type) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::i32, PartialShape{1, 8, 128}),
                                          param(element::i32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::i32, PartialShape{1, 8, 64}),
                                          param(element::i32, PartialShape{1, 8, 64}),
                                          param(element::i32, PartialShape{128}),
                                          param(element::i32, PartialShape{1, 3, 128}),
                                          param(element::i32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("must be a floating point type"));
}

TEST_F(TypePropQSAPrepareTest, mismatched_data_element_types) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f16, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The element type of past_block_index_keys must be the same as of raw_index_keys."));
}

TEST_F(TypePropQSAPrepareTest, incorrect_key_length_element_type) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::f32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The element type of key_length must be i32 or i64."));
}

TEST_F(TypePropQSAPrepareTest, single_token_blocks_are_rejected) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          1),
                    NodeValidationFailure,
                    HasSubstr("The attribute key_block_size must be an integer greater than 1."));
}

TEST_F(TypePropQSAPrepareTest, incorrect_epsilon) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 8, 128}),
                                          param(element::f32, PartialShape{1, 0, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{1, 8, 64}),
                                          param(element::f32, PartialShape{128}),
                                          param(element::f32, PartialShape{1, 3, 128}),
                                          param(element::f32, PartialShape{1, 128}),
                                          4,
                                          0.0f),
                    NodeValidationFailure,
                    HasSubstr("The attribute epsilon must be a positive floating point number."));
}
}  // namespace ov::test
