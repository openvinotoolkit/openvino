// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/augru_sequence.hpp"

#include "gtest/gtest.h"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/opsets/opset9.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ov;

struct augru_sequence_parameters {
    Dimension batch_size = 8;
    Dimension num_directions = 1;
    Dimension seq_length = 6;
    Dimension input_size = 4;
    Dimension hidden_size = 128;
    element::Type et = element::f32;
};

shared_ptr<op::internal::AUGRUSequence> augru_seq_init(const augru_sequence_parameters& params) {
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
        ASSERT_THROW(augru_sequence->validate_and_infer_types(), ngraph::CheckFailure)
            << "GRUSequence node was created with invalid data.";
    }
}

TEST(type_prop, augru_sequence_invalid_input_dynamic_rank) {
    augru_sequence_parameters params;

    params.batch_size = 8;
    params.num_directions = 1;
    params.seq_length = 6;
    params.input_size = 4;
    params.hidden_size = 128;
    params.et = element::f32;

    auto check_dynamic_augru = [](const shared_ptr<op::internal::AUGRUSequence>& augru) -> bool {
        return augru->output(0).get_partial_shape() == PartialShape::dynamic(4) &&
               augru->output(1).get_partial_shape() == PartialShape::dynamic(3) &&
               augru->output(0).get_element_type() == augru->input(0).get_element_type();
    };

    auto augru_sequence = augru_seq_init(params);
    auto invalid_dynamic_tensor = make_shared<opset9::Parameter>(params.et, PartialShape::dynamic());

    // Validate invalid dynamic tensor for all inputs: X, initial_hidden_state, W, R, B, A
    for (size_t i = 0; i < augru_sequence->get_input_size(); i++) {
        augru_sequence = augru_seq_init(params);
        augru_sequence->set_argument(i, invalid_dynamic_tensor);
        augru_sequence->validate_and_infer_types();
        EXPECT_TRUE(check_dynamic_augru(augru_sequence));
    }
}

TEST(type_prop, augru_sequence_invalid_attention_gate) {
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

    try {
        augru_sequence->validate_and_infer_types();
        FAIL() << "AUGRUSequence node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Dimension `seq_length` must be the same for `X` and `A` inputs."));
    }
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
