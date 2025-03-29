// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/augru_sequence.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace testing;

struct augru_sequence_parameters {
    Dimension batch_size = 8;
    Dimension num_directions = 1;
    Dimension seq_length = 6;
    Dimension input_size = 4;
    Dimension hidden_size = 128;
    element::Type et = element::f32;
};

static shared_ptr<op::internal::AUGRUSequence> augru_seq_init(const augru_sequence_parameters& params) {
    auto batch_size = params.batch_size;
    auto seq_length = params.seq_length;
    auto input_size = params.input_size;
    auto num_directions = params.num_directions;
    auto hidden_size = params.hidden_size;
    auto et = params.et;

    const auto X = make_shared<opset9::Parameter>(et, PartialShape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<opset9::Parameter>(et, PartialShape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<opset9::Parameter>(et, PartialShape{batch_size});
    const auto W = make_shared<opset9::Parameter>(et, PartialShape{num_directions, hidden_size * 3, input_size});
    const auto R = make_shared<opset9::Parameter>(et, PartialShape{num_directions, hidden_size * 3, hidden_size});
    const auto B = make_shared<opset9::Parameter>(et, PartialShape{num_directions, hidden_size * 3});
    const auto A = make_shared<opset9::Parameter>(et, PartialShape{batch_size, seq_length, 1});

    const auto augru_sequence = make_shared<op::internal::AUGRUSequence>(
        X,
        initial_hidden_state,
        sequence_lengths,
        W,
        R,
        B,
        A,
        static_cast<size_t>(hidden_size.is_static() ? hidden_size.get_length() : 128));

    return augru_sequence;
}

TEST(type_prop, augru_sequence) {
    augru_sequence_parameters params;
    params.batch_size = 8;
    params.num_directions = 1;
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = 128;

    const auto augru_seq = augru_seq_init(params);

    EXPECT_EQ(augru_seq->get_hidden_size(), static_cast<size_t>(params.hidden_size.get_length()));
    EXPECT_EQ(augru_seq->get_direction(), op::RecurrentSequenceDirection::FORWARD);
    EXPECT_TRUE(augru_seq->get_activations_alpha().empty());
    EXPECT_TRUE(augru_seq->get_activations_beta().empty());
    EXPECT_EQ(augru_seq->get_activations()[0], "sigmoid");
    EXPECT_EQ(augru_seq->get_activations()[1], "tanh");
    EXPECT_EQ(augru_seq->get_clip(), 0.f);
    EXPECT_EQ(augru_seq->get_linear_before_reset(), false);
    EXPECT_EQ(augru_seq->get_output_element_type(0), element::f32);
    EXPECT_EQ(augru_seq->outputs().size(), 2);
    EXPECT_EQ(augru_seq->get_output_partial_shape(0),
              (PartialShape{params.batch_size, params.num_directions, params.seq_length, params.hidden_size}));
    EXPECT_EQ(augru_seq->get_output_element_type(1), element::f32);
    EXPECT_EQ(augru_seq->get_output_partial_shape(1),
              (PartialShape{params.batch_size, params.num_directions, params.hidden_size}));
}

TEST(type_prop, augru_sequence_dynamic_batch_size) {
    augru_sequence_parameters params;
    params.batch_size = Dimension::dynamic();
    params.num_directions = 1;
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = 128;
    params.et = element::f32;

    auto augru_seq = augru_seq_init(params);

    EXPECT_EQ(augru_seq->get_output_partial_shape(0),
              (PartialShape{params.batch_size, params.num_directions, params.seq_length, params.hidden_size}));
    EXPECT_EQ(augru_seq->get_output_partial_shape(1),
              (PartialShape{params.batch_size, params.num_directions, params.hidden_size}));
    EXPECT_EQ(augru_seq->get_output_element_type(0), params.et);
    EXPECT_EQ(augru_seq->get_output_element_type(1), params.et);
}

TEST(type_prop, augru_sequence_dynamic_num_directions) {
    augru_sequence_parameters params;
    params.batch_size = 8;
    params.num_directions = Dimension::dynamic();
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = 128;
    params.et = element::f32;

    auto augru_sequence = augru_seq_init(params);

    EXPECT_EQ(augru_sequence->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_EQ(augru_sequence->get_output_partial_shape(1), (PartialShape{params.batch_size, 1, params.hidden_size}));
    EXPECT_EQ(augru_sequence->get_output_element_type(0), params.et);
    EXPECT_EQ(augru_sequence->get_output_element_type(1), params.et);
}

TEST(type_prop, augru_sequence_dynamic_seq_length) {
    augru_sequence_parameters params;
    params.batch_size = 8;
    params.num_directions = 1;
    params.seq_length = Dimension::dynamic();
    params.input_size = 4;
    params.hidden_size = 128;
    params.et = element::f32;

    auto augru_sequence = augru_seq_init(params);

    EXPECT_EQ(augru_sequence->get_output_partial_shape(0),
              (PartialShape{params.batch_size, params.num_directions, params.seq_length, params.hidden_size}));
    EXPECT_EQ(augru_sequence->get_output_partial_shape(1),
              (PartialShape{params.batch_size, params.num_directions, params.hidden_size}));
    EXPECT_EQ(augru_sequence->get_output_element_type(0), params.et);
    EXPECT_EQ(augru_sequence->get_output_element_type(1), params.et);
}

TEST(type_prop, augru_sequence_dynamic_hidden_size) {
    augru_sequence_parameters params;
    params.batch_size = 8;
    params.num_directions = 1;
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = Dimension::dynamic();
    params.et = element::f32;

    auto augru_sequence = augru_seq_init(params);

    EXPECT_EQ(augru_sequence->get_output_partial_shape(0),
              (PartialShape{params.batch_size, params.num_directions, params.seq_length, params.hidden_size}));
    EXPECT_EQ(augru_sequence->get_output_partial_shape(1),
              (PartialShape{params.batch_size, params.num_directions, params.hidden_size}));
    EXPECT_EQ(augru_sequence->get_output_element_type(0), params.et);
    EXPECT_EQ(augru_sequence->get_output_element_type(1), params.et);
}

TEST(type_prop, augru_sequence_invalid_input_dimension) {
    augru_sequence_parameters params;

    params.batch_size = 8;
    params.num_directions = 1;
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = 128;
    params.et = element::f32;

    auto augru_sequence = augru_seq_init(params);
    auto invalid_rank_tensor = make_shared<opset9::Parameter>(params.et, PartialShape{});

    // Validate invalid rank0 tensor for all inputs: X, initial_hidden_state, W, R, B, A
    for (size_t i = 0; i < augru_sequence->get_input_size(); i++) {
        augru_sequence = augru_seq_init(params);
        augru_sequence->set_argument(i, invalid_rank_tensor);
        ASSERT_THROW(augru_sequence->validate_and_infer_types(), ov::AssertFailure)
            << "AUGRUSequence node was created with invalid data.";
    }
}

TEST(type_prop, augru_sequence_input_dynamic_shape_ranges) {
    augru_sequence_parameters param;

    param.batch_size = Dimension(1, 8);
    param.num_directions = Dimension(1, 2);
    param.seq_length = Dimension(5, 7);
    param.input_size = Dimension(64, 128);
    param.hidden_size = Dimension(32, 64);
    param.et = element::f32;

    auto augru_sequence = augru_seq_init(param);
    augru_sequence->validate_and_infer_types();

    EXPECT_EQ(augru_sequence->get_output_partial_shape(0),
              (PartialShape{param.batch_size, 1, param.seq_length, param.hidden_size}));
    EXPECT_EQ(augru_sequence->get_output_partial_shape(1), (PartialShape{param.batch_size, 1, param.hidden_size}));
    EXPECT_EQ(augru_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(augru_sequence->get_output_element_type(1), param.et);
}

TEST(type_prop, augru_sequence_input_dynamic_rank) {
    augru_sequence_parameters param;

    param.batch_size = 8;
    param.num_directions = 1;
    param.seq_length = 6;
    param.input_size = 4;
    param.hidden_size = 128;
    param.et = element::f32;

    auto augru_sequence = augru_seq_init(param);
    auto dynamic_tensor = make_shared<opset9::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));

    for (size_t i = 0; i < augru_sequence->get_input_size(); i++) {
        augru_sequence = augru_seq_init(param);
        augru_sequence->set_argument(i, dynamic_tensor);
        augru_sequence->validate_and_infer_types();
        if (i == 0) {  // X input dynamic rank
            EXPECT_EQ(augru_sequence->get_output_partial_shape(0),
                      (PartialShape{param.batch_size, param.num_directions, -1, param.hidden_size}));
        } else {
            EXPECT_EQ(augru_sequence->get_output_partial_shape(0),
                      (PartialShape{param.batch_size, param.num_directions, param.seq_length, param.hidden_size}));
        }
        EXPECT_EQ(augru_sequence->get_output_partial_shape(1),
                  (PartialShape{param.batch_size, param.num_directions, param.hidden_size}));
        EXPECT_EQ(augru_sequence->get_output_element_type(0), param.et);
        EXPECT_EQ(augru_sequence->get_output_element_type(1), param.et);
    }
}

TEST(type_prop, augru_sequence_all_inputs_dynamic_rank) {
    augru_sequence_parameters param;

    param.batch_size = 8;
    param.num_directions = 1;
    param.seq_length = 6;
    param.input_size = 4;
    param.hidden_size = 128;
    param.et = element::f32;

    const auto X = make_shared<opset9::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));
    const auto initial_hidden_state = make_shared<opset9::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));
    const auto sequence_lengths = make_shared<opset9::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));
    const auto W = make_shared<opset9::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));
    const auto R = make_shared<opset9::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));
    const auto B = make_shared<opset9::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));
    const auto A = make_shared<opset9::Parameter>(param.et, PartialShape::dynamic(Rank::dynamic()));

    const auto augru_sequence = make_shared<op::internal::AUGRUSequence>(X,
                                                                         initial_hidden_state,
                                                                         sequence_lengths,
                                                                         W,
                                                                         R,
                                                                         B,
                                                                         A,
                                                                         param.hidden_size.get_length());
    EXPECT_EQ(augru_sequence->get_output_partial_shape(0), (PartialShape{-1, 1, -1, -1}));
    EXPECT_EQ(augru_sequence->get_output_partial_shape(1), (PartialShape{-1, 1, -1}));
    EXPECT_EQ(augru_sequence->get_output_element_type(0), param.et);
    EXPECT_EQ(augru_sequence->get_output_element_type(1), param.et);
}

TEST(type_prop, augru_sequence_invalid_attention_gate_seq_length) {
    augru_sequence_parameters params;

    params.batch_size = 8;
    params.num_directions = 1;
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = 128;
    params.et = element::f32;

    auto augru_sequence = augru_seq_init(params);
    auto invalid_attention_gate = make_shared<opset9::Parameter>(params.et, PartialShape{params.batch_size, 999, 1});
    augru_sequence->set_argument(6, invalid_attention_gate);

    OV_EXPECT_THROW(augru_sequence->validate_and_infer_types(),
                    ov::NodeValidationFailure,
                    HasSubstr("Dimension `seq_length` must be the same for `X` and `A` inputs"));
}

TEST(type_prop, augru_sequence_invalid_attention_gate_batch) {
    augru_sequence_parameters params;

    params.batch_size = 8;
    params.num_directions = 1;
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = 128;
    params.et = element::f32;

    auto augru_sequence = augru_seq_init(params);
    auto invalid_attention_gate = make_shared<opset9::Parameter>(params.et, PartialShape{999, params.seq_length, 1});
    augru_sequence->set_argument(6, invalid_attention_gate);

    OV_EXPECT_THROW(augru_sequence->validate_and_infer_types(),
                    ov::NodeValidationFailure,
                    HasSubstr("Dimension `batch_size` must be the same for `X` and `A` inputs"));
}

namespace {
struct NotSupportedArguments {
    float clip = 0;
    vector<string> activations = {"sigmoid", "tanh"};
    vector<float> activations_alpha, activations_beta;
    string direction = "forward";
    bool linear_before_reset = false;
};

class AttributeVisitorMock : public AttributeVisitor {
public:
    AttributeVisitorMock(const NotSupportedArguments& args) : m_args{args} {}
    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override {}

    void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override {
        if ("clip" == name) {
            adapter.set(m_args.clip);
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<vector<string>>& adapter) override {
        if ("activations" == name) {
            adapter.set(m_args.activations);
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<vector<float>>& adapter) override {
        if ("activations_alpha" == name) {
            adapter.set(m_args.activations_alpha);
        }
        if ("activations_beta" == name) {
            adapter.set(m_args.activations_beta);
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<string>& adapter) override {
        if ("direction" == name) {
            adapter.set(m_args.direction);
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override {
        if ("linear_before_reset" == name) {
            adapter.set(m_args.linear_before_reset);
        }
    }

private:
    NotSupportedArguments m_args;
};
}  // namespace

TEST(type_prop, augru_sequence_not_supported_attributes) {
    augru_sequence_parameters params;

    params.batch_size = 8;
    params.num_directions = 1;
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = 128;
    params.et = element::f32;
    auto augru_sequence = augru_seq_init(params);

    NotSupportedArguments args;
    args.clip = 2.f;
    AttributeVisitorMock visitor(args);
    augru_sequence->visit_attributes(visitor);

    try {
        augru_sequence->validate_and_infer_types();
        FAIL() << "AUGRUSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("AUGRUSequence doesn't support clip other than 0."));
    }

    args = NotSupportedArguments();
    args.activations = {"relu", "tanh"};
    visitor = AttributeVisitorMock(args);
    augru_sequence->visit_attributes(visitor);

    try {
        augru_sequence->validate_and_infer_types();
        FAIL() << "AUGRUSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("AUGRUSequence supports only sigmoid for f and tanh for g activation functions."));
    }

    args = NotSupportedArguments();
    args.activations_beta = {1, 2};
    visitor = AttributeVisitorMock(args);
    augru_sequence->visit_attributes(visitor);

    try {
        augru_sequence->validate_and_infer_types();
        FAIL() << "AUGRUSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("AUGRUSequence doesn't support activations_alpha and activations_beta."));
    }

    args = NotSupportedArguments();
    args.direction = "reverse";
    visitor = AttributeVisitorMock(args);
    augru_sequence->visit_attributes(visitor);

    try {
        augru_sequence->validate_and_infer_types();
        FAIL() << "AUGRUSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("AUGRUSequence supports only forward direction."));
    }

    args = NotSupportedArguments();
    args.linear_before_reset = true;
    visitor = AttributeVisitorMock(args);
    augru_sequence->visit_attributes(visitor);

    try {
        augru_sequence->validate_and_infer_types();
        FAIL() << "AUGRUSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("AUGRUSequence supports only linear_before_reset equals false."));
    }
}
