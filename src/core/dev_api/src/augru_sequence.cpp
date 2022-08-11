// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "augru_sequence.hpp"

#include <memory>
#include <string>
#include <vector>

#include "itt.hpp"

using namespace std;

namespace ov {
    BWDCMP_RTTI_DEFINITION(op::v1::AUGRUSequence);

    op::v1::AUGRUSequence::AUGRUSequence()
        : m_direction(op::RecurrentSequenceDirection::FORWARD),
        m_linear_before_reset(false) {}

    op::v1::AUGRUSequence::AUGRUSequence(const Output<Node>& X,
                                        const Output<Node>& H_t,
                                        const Output<Node>& sequence_lengths,
                                        const Output<Node>& W,
                                        const Output<Node>& R,
                                        const Output<Node>& B,
                                        const Output<Node>& A,
                                        std::size_t hidden_size)
        : RNNCellBase({X, H_t, sequence_lengths, W, R, B, A},
                    hidden_size,
                    0.f,
                    std::vector<std::string>{"sigmoid", "tanh"},
                    {},
                    {}),
        m_direction(op::RecurrentSequenceDirection::FORWARD),
        m_linear_before_reset(false) {
        constructor_validate_and_infer_types();
    }

    void op::v1::AUGRUSequence::validate_and_infer_types() {
        NGRAPH_OP_SCOPE(v1_AUGRUSequence_validate_and_infer_types);
        for (const auto& input : inputs()) {
            if (input.get_partial_shape().rank().is_dynamic()) {
                set_output_type(0, get_input_element_type(0), ov::PartialShape::dynamic(4));
                set_output_type(1, get_input_element_type(0), ov::PartialShape::dynamic(3));
                return;
            }
        }

        auto gru_seq_gates_count = 3;
        auto merged_batch_size = Dimension::dynamic();
        auto merged_hidden_size = Dimension::dynamic();
        auto merged_num_directions = Dimension::dynamic();
        auto result_et = element::dynamic;

        auto x_pshape = get_input_partial_shape(0);
        auto ht_pshape = get_input_partial_shape(1);
        auto sl_pshape = get_input_partial_shape(2);
        auto w_pshape = get_input_partial_shape(3);
        auto r_pshape = get_input_partial_shape(4);
        auto b_pshape = get_input_partial_shape(5);
        auto a_pshape = get_input_partial_shape(6);

       validate_seq_input_rank_dimension(
            {x_pshape, ht_pshape, sl_pshape, w_pshape, r_pshape, b_pshape, a_pshape});

        // Validate input types and save result for output type
        NODE_VALIDATION_CHECK(this,
                            element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                                element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                                element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                                element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                                element::Type::merge(result_et, result_et, get_input_element_type(5)) &&
                                element::Type::merge(result_et, result_et, get_input_element_type(6)),
                            "Element types for inputs do not match.");

        // Merge batch_size dimension across all inputs to evaluate output[0] dimension
        NODE_VALIDATION_CHECK(this,
                            Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                                Dimension::merge(merged_batch_size, merged_batch_size, a_pshape[0]) &&
                                Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]) &&
                                Dimension::merge(merged_batch_size, merged_batch_size, sl_pshape[0]),
                            "Dimension batch_size is not matched between inputs.");

        // A input shape validation // [batch_size, seq_length, 1]
        NODE_VALIDATION_CHECK(this, a_pshape.rank().compatible(3), "'A' input must be a 3D tensor.");
        if (a_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this,
                                a_pshape[1].compatible(x_pshape[1]),
                                "Dimension `seq_length` must be the same for `X` and `A` inputs.");
            NODE_VALIDATION_CHECK(this, a_pshape[2].compatible(1), "The last dimension of `A` shape must be equal to `1`.");
        }

        // Merge hidden_size dimension across all inputs to evaluate output dimension
        NODE_VALIDATION_CHECK(this,
                            Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[2]) &&
                                Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[2]),
                            "Parameter hidden_size not matched AUGRUSequence.");

        // Merge num_directions dimension across all inputs to evaluate output dimension
        NODE_VALIDATION_CHECK(this,
                            Dimension::merge(merged_num_directions, merged_num_directions, ht_pshape[1]) &&
                                Dimension::merge(merged_num_directions, merged_num_directions, w_pshape[0]) &&
                                Dimension::merge(merged_num_directions, merged_num_directions, r_pshape[0]) &&
                                Dimension::merge(merged_num_directions, merged_num_directions, b_pshape[0]),
                            "Parameter num_directions not matched in AUGRUSequence.");

        auto valid_num_directions = 1;  // AUGRUSequence supports only Forward direction
        NODE_VALIDATION_CHECK(this,
                            Dimension::merge(merged_num_directions, merged_num_directions, valid_num_directions),
                            "Parameter 'num_directions' doesn't match with direction '",
                            m_direction,
                            "' in AUGRUSequence. Expected ",
                            valid_num_directions,
                            ", actual ",
                            merged_num_directions);

        // Validate hidden_size value for W, R, B inputs
        if (merged_hidden_size.is_static()) {
            if (w_pshape[1].is_static()) {
                NODE_VALIDATION_CHECK(this,
                                    w_pshape[1].compatible(merged_hidden_size * gru_seq_gates_count),
                                    "Parameter hidden_size mistmatched in W input. Current value is: ",
                                    w_pshape[1].get_length(),
                                    ", expected: ",
                                    merged_hidden_size.get_length() * gru_seq_gates_count,
                                    ".");
            }

            if (r_pshape[1].is_static()) {
                NODE_VALIDATION_CHECK(this,
                                    r_pshape[1].compatible(merged_hidden_size * gru_seq_gates_count),
                                    "Parameter hidden_size mistmatched in R input. Current value is: ",
                                    r_pshape[1].get_length(),
                                    ", expected: ",
                                    merged_hidden_size.get_length() * gru_seq_gates_count,
                                    ".");
            }

            if (b_pshape[1].is_static()) {
                NODE_VALIDATION_CHECK(
                    this,
                    b_pshape[1].compatible(merged_hidden_size *
                                        (m_linear_before_reset ? (gru_seq_gates_count + 1) : gru_seq_gates_count)),
                    "Parameter hidden_size mistmatched in B input. Current value is: ",
                    b_pshape[1].get_length(),
                    ", expected: ",
                    merged_hidden_size.get_length() *
                        (m_linear_before_reset ? (gru_seq_gates_count + 1) : gru_seq_gates_count),
                    ".");
            }
        }

        // Mark inputs which are relevant to output parameters
        for (size_t i = 0; i <= 6; ++i) {
            set_input_is_relevant_to_shape(i);
        }

        // Set output size, type and shape
        set_output_size(2);
        set_output_type(0, result_et, {merged_batch_size, merged_num_directions, x_pshape[1], merged_hidden_size});
        set_output_type(1, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
    }

    bool op::v1::AUGRUSequence::visit_attributes(AttributeVisitor& visitor) {
        NGRAPH_OP_SCOPE(v1_AUGRUSequence_visit_attributes);
        visitor.on_attribute("direction", m_direction);
        visitor.on_attribute("linear_before_reset", m_linear_before_reset);
        return op::util::RNNCellBase::visit_attributes(visitor);
    }

    shared_ptr<Node> op::v1::AUGRUSequence::clone_with_new_inputs(const OutputVector& new_args) const {
        NGRAPH_OP_SCOPE(v1_AUGRUSequence_clone_with_new_inputs);
        check_new_args_count(this, new_args);
        return make_shared<op::v1::AUGRUSequence>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                new_args.at(4),
                                                new_args.at(5),
                                                new_args.at(6),
                                                get_hidden_size());
    }
}
