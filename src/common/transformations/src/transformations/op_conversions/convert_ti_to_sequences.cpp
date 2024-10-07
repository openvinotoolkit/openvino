// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_ti_to_sequences.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reverse_sequence.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass;
using namespace ov::op::util;

namespace {
bool convertTensorIteratorToSequence(const std::shared_ptr<ov::op::v0::TensorIterator>& ti,
                                     const std::shared_ptr<ov::op::util::RNNCellBase>& found_cell,
                                     const ov::Output<ov::Node>& data,
                                     const ov::Output<ov::Node>& h_pattern,
                                     const ov::Output<ov::Node>& c_pattern,
                                     const ov::Output<ov::Node>& w_pattern,
                                     const ov::Output<ov::Node>& r_pattern,
                                     const ov::Output<ov::Node>& b_pattern,
                                     const ov::Output<ov::Node>& unsqueeze_after_cell) {
    const auto& func = ti->get_function();
    const auto& params = func->get_parameters();

    std::vector<std::shared_ptr<ov::op::v0::TensorIterator::InputDescription>> ordered_in_descs(3);
    int64_t stride = 0, slice_axis = 0;

    // Remember the order of the X and initial_hidden_state (+ initial_cell_state in case of LSTM) in the TensorIterator
    // params
    for (const auto& input_desc : ti->get_input_descriptions()) {
        auto param = params[input_desc->m_body_parameter_index];
        if (param == data.get_node_shared_ptr()) {
            auto slice_input = std::dynamic_pointer_cast<ov::op::v0::TensorIterator::SliceInputDescription>(input_desc);
            if (!slice_input)
                return false;

            stride = slice_input->m_stride;
            slice_axis = slice_input->m_axis;

            if (!(slice_axis == 0 || slice_axis == 1)) {
                return false;
            }
            ordered_in_descs[0] = input_desc;
        } else if (param == h_pattern.get_node_shared_ptr()) {
            ordered_in_descs[1] = input_desc;
        } else if (param == c_pattern.get_node_shared_ptr()) {
            ordered_in_descs[2] = input_desc;
        } else {
            return false;
        }
    }

    const auto& results = func->get_results();
    std::vector<std::shared_ptr<ov::op::v0::TensorIterator::OutputDescription>> ordered_out_descs(3);

    // Remember the order of cell outputs in the TensorIterator
    for (const auto& output_desc : ti->get_output_descriptions()) {
        std::shared_ptr<ov::op::v0::Result> res = results[output_desc->m_body_value_index];
        if (res->input_value(0) == unsqueeze_after_cell) {
            auto concat_output =
                std::dynamic_pointer_cast<ov::op::v0::TensorIterator::ConcatOutputDescription>(output_desc);
            if (!concat_output)
                return false;

            stride = concat_output->m_stride;
            ordered_out_descs[0] = output_desc;
        } else if (res->input_value(0) == found_cell->output(0)) {
            ordered_out_descs[1] = output_desc;
        } else if (found_cell->get_output_size() == 2 && res->input_value(0) == found_cell->output(1)) {
            ordered_out_descs[2] = output_desc;
        } else {
            return false;
        }
    }

    const auto ti_inputs = ti->input_values();
    auto X = ti_inputs[ordered_in_descs[0]->m_input_index];
    if (slice_axis == 0) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 0, 2});
        X = std::make_shared<ov::op::v1::Transpose>(ti_inputs[ordered_in_descs[0]->m_input_index], order);
    }

    // We must prepare cell inputs to sequence creation: insert num_directions elem via unsqueeze where needed (please,
    // see specification)
    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto initial_hidden_state =
        std::make_shared<ov::op::v0::Unsqueeze>(ti_inputs[ordered_in_descs[1]->m_input_index], axis_1);

    // LSTM case
    std::shared_ptr<ov::Node> initial_cell_state =
        c_pattern.get_node_shared_ptr() == nullptr
            ? nullptr
            : std::make_shared<ov::op::v0::Unsqueeze>(ti_inputs[ordered_in_descs[2]->m_input_index], axis_1);

    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
    auto batch_dimension =
        std::make_shared<ov::op::v1::Gather>(shape_of,
                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                             ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto seq_len_dim = std::make_shared<ov::op::v1::Gather>(shape_of,
                                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                            ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto seq_lengths = std::make_shared<ov::op::v3::Broadcast>(seq_len_dim, batch_dimension);
    auto axis_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto W = ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(w_pattern, axis_0);
    auto R = ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(r_pattern, axis_0);
    auto B = ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(b_pattern, axis_0);

    std::shared_ptr<ov::Node> sequence;
    if (ov::is_type<ov::op::v4::LSTMCell>(found_cell) || ov::is_type<ov::op::v0::LSTMCell>(found_cell)) {
        sequence = std::make_shared<ov::op::v5::LSTMSequence>(
            X,
            initial_hidden_state,
            initial_cell_state,
            seq_lengths,
            W,
            R,
            B,
            found_cell->get_hidden_size(),
            stride > 0 ? ov::op::RecurrentSequenceDirection::FORWARD : ov::op::RecurrentSequenceDirection::REVERSE,
            found_cell->get_activations_alpha(),
            found_cell->get_activations_beta(),
            found_cell->get_activations(),
            found_cell->get_clip());
    } else if (ov::is_type<ov::op::v0::RNNCell>(found_cell)) {
        sequence = std::make_shared<ov::op::v5::RNNSequence>(
            X,
            initial_hidden_state,
            seq_lengths,
            W,
            R,
            B,
            found_cell->get_hidden_size(),
            stride > 0 ? ov::op::RecurrentSequenceDirection::FORWARD : ov::op::RecurrentSequenceDirection::REVERSE,
            found_cell->get_activations(),
            found_cell->get_activations_alpha(),
            found_cell->get_activations_beta(),
            found_cell->get_clip());
    } else if (ov::is_type<ov::op::v3::GRUCell>(found_cell)) {
        const auto gru_cell = ov::as_type_ptr<ov::op::v3::GRUCell>(found_cell);
        sequence = std::make_shared<ov::op::v5::GRUSequence>(
            X,
            initial_hidden_state,
            seq_lengths,
            W,
            R,
            B,
            gru_cell->get_hidden_size(),
            stride > 0 ? ov::op::RecurrentSequenceDirection::FORWARD : ov::op::RecurrentSequenceDirection::REVERSE,
            gru_cell->get_activations(),
            gru_cell->get_activations_alpha(),
            gru_cell->get_activations_beta(),
            gru_cell->get_clip(),
            gru_cell->get_linear_before_reset());
    } else {
        OPENVINO_THROW("Unsupported sequence type");
    }

    ov::Output<ov::Node> out = sequence->output(0);
    if (slice_axis == 0) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {2, 1, 0, 3});
        out = std::make_shared<ov::op::v1::Transpose>(out, order);
    }

    ov::NodeVector outputs;
    // We must remove num_directions dimension that was added before sequence creation
    auto axis_out = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto out_0 = std::make_shared<ov::op::v0::Squeeze>(out, axis_out);
    auto out_1 = std::make_shared<ov::op::v0::Squeeze>(sequence->output(1), axis_out);
    out_0->set_friendly_name(ti->get_friendly_name() + ".0");
    out_1->set_friendly_name(ti->get_friendly_name() + ".1");
    outputs.emplace_back(out_0);
    outputs.emplace_back(out_1);

    if (sequence->get_output_size() == 3) {
        auto out_2 = std::make_shared<ov::op::v0::Squeeze>(sequence->output(2), axis_out);
        out_2->set_friendly_name(ti->get_friendly_name() + ".2");
        outputs.emplace_back(out_2);
    }

    for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
        if (ordered_out_descs[i]) {
            ti->output(ordered_out_descs[i]->m_output_index).replace(outputs[i]->output(0));
        }
    }

    ov::NodeVector new_nodes = outputs;
    new_nodes.emplace_back(initial_hidden_state);
    new_nodes.emplace_back(W);
    new_nodes.emplace_back(R);
    new_nodes.emplace_back(B);
    new_nodes.emplace_back(sequence);

    if (c_pattern.get_node_shared_ptr()) {
        new_nodes.emplace_back(initial_cell_state);
    }

    new_nodes.emplace_back(batch_dimension);
    new_nodes.emplace_back(shape_of);
    new_nodes.emplace_back(seq_len_dim);
    new_nodes.emplace_back(seq_lengths);

    if (slice_axis == 0) {
        new_nodes.emplace_back(out.get_node_shared_ptr());
        new_nodes.emplace_back(X.get_node_shared_ptr());
    }

    copy_runtime_info(ti, new_nodes);
    return true;
}

bool check_condition_increment_pattern(
    const std::shared_ptr<op::v0::Result>& cond_result,
    const std::shared_ptr<op::v5::Loop>& loop,
    const ov::ParameterVector& body_params,
    const ov::ResultVector& body_results,
    const ov::op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector& input_descriptions,
    Output<Node>& num_iters_output) {
    // return a number of iterations if pattern matches
    // pattern for condition sub-graph in Loop operarion
    auto num_iter_const_label = pattern::wrap_type<op::v0::Constant>();
    auto num_iter_param_label = pattern::wrap_type<op::v0::Parameter>();
    auto num_iterations_label =
        std::make_shared<pattern::op::Or>(OutputVector{num_iter_const_label, num_iter_param_label});
    auto counter_label = pattern::wrap_type<op::v0::Parameter>();
    auto counter_step_label = pattern::wrap_type<op::v0::Constant>();
    auto updated_counter_label = pattern::wrap_type<op::v1::Add>({counter_label, counter_step_label});
    auto less_label = pattern::wrap_type<op::v1::Less>({updated_counter_label, num_iterations_label});
    auto condition_label = pattern::wrap_type<op::v0::Result>({less_label});

    // check a pattern of condition graph and get a number of iterations
    ov::pass::pattern::Matcher condition_matcher(condition_label);
    if (!condition_matcher.match(cond_result->output(0))) {
        return false;
    }
    const auto& condition_map = condition_matcher.get_pattern_value_map();
    int64_t counter_step = -1;
    if (!get_constant_value(condition_map.at(counter_step_label).get_node_shared_ptr(), counter_step) ||
        counter_step != 1) {
        return false;
    }
    // get initial value of counter
    int64_t initial_counter = -1;
    auto counter_param = ov::as_type_ptr<ov::op::v0::Parameter>(condition_map.at(counter_label).get_node_shared_ptr());
    if (!counter_param) {
        return false;
    }
    for (const auto& input_desc : input_descriptions) {
        auto body_param_idx = input_desc->m_body_parameter_index;
        auto input_idx = input_desc->m_input_index;
        if (body_params[body_param_idx] != counter_param) {
            continue;
        }
        // it must be merged input and incremented each iteration
        auto merged_desc = ov::as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(input_desc);
        if (!merged_desc) {
            return false;
        }
        auto result_idx = merged_desc->m_body_value_index;
        auto update_counter = body_results[result_idx]->input_value(0).get_node_shared_ptr();
        if (update_counter != condition_map.at(updated_counter_label).get_node_shared_ptr()) {
            return false;
        }
        // get initial value of counter
        if (!get_constant_value(loop->input_value(input_idx).get_node_shared_ptr(), initial_counter)) {
            return false;
        }
        // suitable counter-parameter is found and checked
        break;
    }
    if (initial_counter != 0) {
        return false;
    }
    // retrieve a number of iterations
    int64_t num_iters = -1;
    if (condition_map.count(num_iter_param_label)) {
        auto num_iter_param = condition_map.at(num_iter_param_label).get_node_shared_ptr();
        for (const auto& input_desc : input_descriptions) {
            auto body_param_idx = input_desc->m_body_parameter_index;
            auto input_idx = input_desc->m_input_index;
            if (body_params[body_param_idx] != num_iter_param) {
                continue;
            }
            num_iters_output = loop->input_value(input_idx);
            break;
        }
    } else if (condition_map.count(num_iter_const_label)) {
        if (!get_constant_value(condition_map.at(num_iter_const_label).get_node_shared_ptr(), num_iters) ||
            num_iters < 1) {
            return false;
        }
        num_iters_output = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, num_iters);
    } else {
        return false;
    }

    return true;
}

bool check_condition_true_pattern(const std::shared_ptr<op::v0::Result>& cond_result,
                                  const std::shared_ptr<op::v5::Loop>& loop,
                                  Output<Node>& num_iters_output) {
    // return a number of iterations if pattern matches
    auto cond_const_label = pattern::wrap_type<op::v0::Constant>();
    auto condition_label = pattern::wrap_type<op::v0::Result>({cond_const_label});

    // check a pattern of condition graph and get a number of iterations
    ov::pass::pattern::Matcher condition_matcher(condition_label);
    if (!condition_matcher.match(cond_result->output(0))) {
        return false;
    }
    const auto& condition_map = condition_matcher.get_pattern_value_map();
    const auto& cond_const =
        ov::as_type_ptr<op::v0::Constant>(condition_map.at(cond_const_label).get_node_shared_ptr());
    bool cond_value = false;
    if (!ov::op::util::get_constant_value(cond_const, cond_value) || !cond_value) {
        return false;
    }
    // number of iteration is retrieve from the first input port
    num_iters_output = loop->input_value(0);

    return true;
}

void create_mul_by_one_pattern(std::shared_ptr<Node>& param,
                               std::shared_ptr<Node>& broadcast_value,
                               std::shared_ptr<Node>& mul,
                               bool with_first_squeeze = false) {
    param = pattern::wrap_type<op::v0::Parameter>();
    auto shape_of_input = param;
    if (with_first_squeeze) {
        auto first_squeeze_axes = pattern::wrap_type<op::v0::Constant>();
        shape_of_input = pattern::wrap_type<op::v0::Squeeze>({param, first_squeeze_axes});
    }
    auto shape_of = pattern::wrap_type<op::v3::ShapeOf>({shape_of_input});
    auto concat_value = pattern::wrap_type<op::v0::Constant>();
    auto concat = pattern::wrap_type<op::v0::Concat>({concat_value, shape_of});
    broadcast_value = pattern::wrap_type<op::v0::Constant>();
    auto broadcast = pattern::wrap_type<op::v3::Broadcast>({broadcast_value, concat});
    auto squeeze_axes = pattern::wrap_type<op::v0::Constant>();
    auto squeeze = pattern::wrap_type<op::v0::Squeeze>({broadcast, squeeze_axes});
    mul = pattern::wrap_type<op::v1::Multiply>({shape_of_input, squeeze});
}

bool check_const_one(const std::shared_ptr<Node>& node) {
    auto const_node = ov::as_type_ptr<op::v0::Constant>(node);
    if (!const_node) {
        return false;
    }
    auto const_values = const_node->get_vector<float>();
    if (const_values.size() != 1 || const_values[0] != 1) {
        return false;
    }
    return true;
}

bool check_lstm_cell_pattern(
    const std::shared_ptr<op::v5::Loop>& loop,
    const ov::ParameterVector& body_params,
    const ov::ResultVector& body_results,
    const ov::op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector& input_descriptions,
    ov::Output<Node>& init_hidden_state,
    ov::Output<Node>& init_cell_state,
    ov::Output<Node>& x,
    ov::Output<Node>& W,
    ov::Output<Node>& R,
    ov::Output<Node>& B,
    std::shared_ptr<op::v0::Result>& result_hidden_state,
    std::shared_ptr<ov::op::v4::LSTMCell>& lstm_cell_node) {
    // check pattern with LSMCell and return key points
    // required for fusion to LSTMSequence
    // pattern for LSTMCell in the body
    auto xi_label = pattern::wrap_type<op::v0::Parameter>();
    auto squeeze_axis_label = pattern::wrap_type<op::v0::Constant>();
    auto xi_reshape_label = pattern::wrap_type<op::v0::Squeeze>({xi_label, squeeze_axis_label});
    auto init_hidden_state_i_label = pattern::wrap_type<op::v0::Parameter>();
    auto init_cell_state_i_label = pattern::wrap_type<op::v0::Parameter>();
    auto W_label = pattern::wrap_type<op::v0::Constant>();
    auto R_label = pattern::wrap_type<op::v0::Constant>();
    auto B_label = pattern::wrap_type<op::v0::Constant>();

    // TODO: 152648 - remove this WA
    // create multiply by one pattern for x input
    std::shared_ptr<Node> x_param, x_broadcast_value, x_mul;
    create_mul_by_one_pattern(x_param, x_broadcast_value, x_mul, true);
    // create multiply by one pattern for hidden state input
    std::shared_ptr<Node> h_param, h_broadcast_value, h_mul;
    create_mul_by_one_pattern(h_param, h_broadcast_value, h_mul, false);

    auto xi_common_label = std::make_shared<pattern::op::Or>(OutputVector{xi_reshape_label, x_mul});
    auto hidden_common_label = std::make_shared<pattern::op::Or>(OutputVector{init_hidden_state_i_label, h_mul});

    auto lstm_cell_label = pattern::wrap_type<op::v4::LSTMCell>(
        {xi_common_label, hidden_common_label, init_cell_state_i_label, W_label, R_label, B_label});
    auto unsqueeze_axis_label = pattern::wrap_type<op::v0::Constant>();
    auto unsqueeze_hidden_state_label = pattern::wrap_type<op::v0::Unsqueeze>({lstm_cell_label, unsqueeze_axis_label});
    auto result_hidden_state_label = pattern::wrap_type<op::v0::Result>({unsqueeze_hidden_state_label});

    // check that body-graph contains a pattern corresponding LSTMCell
    ov::pass::pattern::Matcher lstm_cell_matcher(result_hidden_state_label);
    int64_t unsqueeze_hidden_state_result_idx = -1;
    for (int64_t result_idx = 0; result_idx < static_cast<int64_t>(body_results.size()); ++result_idx) {
        if (lstm_cell_matcher.match(body_results[result_idx]->output(0))) {
            unsqueeze_hidden_state_result_idx = result_idx;
            auto reshape_node =
                lstm_cell_matcher.get_pattern_value_map()[unsqueeze_hidden_state_label].get_node_shared_ptr();
            break;
        }
    }
    if (unsqueeze_hidden_state_result_idx < 0) {
        // not found LSTMCell inside Loop operation
        return false;
    }
    const auto& lstm_cell_map = lstm_cell_matcher.get_pattern_value_map();

    // check that Results with hidden and cell states connected with corresponding
    // Parameter nodes with back edges and found initial hidden and cell states
    lstm_cell_node = ov::as_type_ptr<op::v4::LSTMCell>(lstm_cell_map.at(lstm_cell_label).get_node_shared_ptr());
    if (!lstm_cell_node) {
        return false;
    }
    result_hidden_state =
        ov::as_type_ptr<op::v0::Result>(lstm_cell_map.at(result_hidden_state_label).get_node_shared_ptr());
    if (!result_hidden_state) {
        return false;
    }

    // re-assign keys for pattern if multiply-by-one pattern was applied
    if (lstm_cell_map.count(h_param)) {
        // check that broadcasted value is equal to one
        if (!check_const_one(lstm_cell_map.at(h_broadcast_value).get_node_shared_ptr())) {
            return false;
        }
        init_hidden_state_i_label = h_param;
    }
    if (lstm_cell_map.count(x_param)) {
        // check that broadcasted value is equal to one
        if (!check_const_one(lstm_cell_map.at(x_broadcast_value).get_node_shared_ptr())) {
            return false;
        }
        xi_label = x_param;
    }

    for (const auto& input_desc : input_descriptions) {
        auto param_idx = input_desc->m_body_parameter_index;
        auto input_idx = input_desc->m_input_index;
        if (body_params[param_idx] == lstm_cell_map.at(init_hidden_state_i_label).get_node_shared_ptr()) {
            // hidden state Parameter node
            auto merged_desc = ov::as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(input_desc);
            if (!merged_desc) {
                return false;
            }
            const auto& hidden_state_result = body_results[merged_desc->m_body_value_index];
            if ((hidden_state_result->get_input_node_shared_ptr(0) != lstm_cell_node) ||
                (hidden_state_result->input_value(0).get_index() != 0)) {
                return false;
            }
            init_hidden_state = loop->input_value(input_idx);
        } else if (body_params[param_idx] == lstm_cell_map.at(init_cell_state_i_label).get_node_shared_ptr()) {
            // cell state Parameter node
            auto merged_desc = ov::as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(input_desc);
            if (!merged_desc) {
                return false;
            }
            const auto& cell_state_result = body_results[merged_desc->m_body_value_index];
            if ((cell_state_result->get_input_node_shared_ptr(0) != lstm_cell_node) ||
                (cell_state_result->input_value(0).get_index() != 1)) {
                return false;
            }
            init_cell_state = loop->input_value(input_idx);
        } else if (body_params[param_idx] == lstm_cell_map.at(xi_label).get_node_shared_ptr()) {
            // input data Parameter node
            auto sliced_desc = ov::as_type_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>(input_desc);
            if (!sliced_desc || (sliced_desc->m_axis != 0) || (sliced_desc->m_start != 0) ||
                (sliced_desc->m_stride != 1) || (sliced_desc->m_end != -1)) {
                return false;
            }
            x = loop->input_value(input_idx);
        }
    }

    W = lstm_cell_map.at(W_label);
    R = lstm_cell_map.at(R_label);
    B = lstm_cell_map.at(B_label);

    return true;
}
}  // namespace

ov::pass::ConvertTensorIteratorToLSTMSequence::ConvertTensorIteratorToLSTMSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToLSTMSequence);
    auto tensor_iterator = pattern::wrap_type<ov::op::v0::TensorIterator>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(3));
        auto pattern_1 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto squeeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Squeeze>({data, pattern_1});

        auto input_H_state = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(2));
        auto input_C_state = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(2));
        auto input_W = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_R = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_B = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));

        ov::OutputVector cell_inputs{squeeze, input_H_state, input_C_state, input_W, input_R, input_B};
        auto cell = ov::pass::pattern::wrap_type<ov::op::v0::LSTMCell, ov::op::v4::LSTMCell>(cell_inputs);

        auto pattern_2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto unsqueeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({cell, pattern_2});
        ov::pass::pattern::Matcher matcher(unsqueeze);

        bool match = false;
        auto func = ti->get_body();
        for (const auto& res : func->get_results()) {
            match = matcher.match((res->get_input_source_output(0)));
            if (match)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!match || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        const auto& pattern_map = matcher.get_pattern_value_map();
        std::shared_ptr<Node> found_cell = pattern_map.at(cell).get_node_shared_ptr();
        const auto lstm_cell = std::dynamic_pointer_cast<ov::op::util::RNNCellBase>(found_cell);
        if (lstm_cell == nullptr)
            return false;

        return convertTensorIteratorToSequence(ti,
                                               lstm_cell,
                                               pattern_map.at(data),
                                               pattern_map.at(input_H_state),
                                               pattern_map.at(input_C_state),
                                               pattern_map.at(input_W),
                                               pattern_map.at(input_R),
                                               pattern_map.at(input_B),
                                               pattern_map.at(unsqueeze));
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertTensorIteratorToRNNSequence::ConvertTensorIteratorToRNNSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToRNNSequence);
    auto tensor_iterator = pattern::wrap_type<ov::op::v0::TensorIterator>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(3));
        auto pattern_1 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto squeeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Squeeze>({data, pattern_1});

        auto input_H_state = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(2));
        auto input_W = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_R = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_B = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));

        ov::OutputVector cell_inputs{squeeze, input_H_state, input_W, input_R, input_B};
        auto cell = ov::pass::pattern::wrap_type<ov::op::v0::RNNCell>(cell_inputs);

        auto pattern_2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto unsqueeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({cell, pattern_2});
        ov::pass::pattern::Matcher matcher(unsqueeze);

        bool match = false;
        auto func = ti->get_body();
        for (const auto& res : func->get_results()) {
            match = matcher.match((res->get_input_source_output(0)));
            if (match)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!match || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto& rnn_cell = ov::as_type_ptr<ov::op::v0::RNNCell>(pattern_map.at(cell).get_node_shared_ptr());
        if (rnn_cell == nullptr)
            return false;

        return convertTensorIteratorToSequence(ti,
                                               rnn_cell,
                                               pattern_map.at(data),
                                               pattern_map.at(input_H_state),
                                               ov::Output<ov::Node>(),
                                               pattern_map.at(input_W),
                                               pattern_map.at(input_R),
                                               pattern_map.at(input_B),
                                               pattern_map.at(unsqueeze));
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertTensorIteratorToGRUSequence::ConvertTensorIteratorToGRUSequence() {
    MATCHER_SCOPE(ConvertTensorIteratorToGRUSequence);
    auto tensor_iterator = pattern::wrap_type<ov::op::v0::TensorIterator>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(m.get_match_root());
        if (!ti || transformation_callback(ti))
            return false;

        // create a pattern for the TensorIterator body
        auto data = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(3));
        auto pattern_1 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto squeeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Squeeze>({data, pattern_1});

        auto input_H_state = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::rank_equals(2));
        auto input_W = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_R = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(2));
        auto input_B = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));

        ov::OutputVector cell_inputs{squeeze, input_H_state, input_W, input_R, input_B};
        auto cell = ov::pass::pattern::wrap_type<ov::op::v3::GRUCell>(cell_inputs);

        auto pattern_2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
        auto unsqueeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({cell, pattern_2});

        ov::pass::pattern::Matcher matcher(unsqueeze);

        bool match = false;
        auto func = ti->get_body();
        for (const auto& res : func->get_results()) {
            match = matcher.match((res->get_input_source_output(0)));
            if (match)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!match || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto& gru_cell = ov::as_type_ptr<ov::op::v3::GRUCell>(pattern_map.at(cell).get_node_shared_ptr());
        if (gru_cell == nullptr)
            return false;

        return convertTensorIteratorToSequence(ti,
                                               gru_cell,
                                               pattern_map.at(data),
                                               pattern_map.at(input_H_state),
                                               ov::Output<ov::Node>(),
                                               pattern_map.at(input_W),
                                               pattern_map.at(input_R),
                                               pattern_map.at(input_B),
                                               pattern_map.at(unsqueeze));
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(tensor_iterator, matcher_name);
    register_matcher(m, callback);
}

// clang-format off
/*

   Following subgraph in Loop is fused into LSTMSequence


   +------------------------------+
   |              X               |    +----------------+    +------+
   |         (invariant)          |    | sequence index |    | axis |
   | [seq_len, batch, input_size] |    |       []       |    | {0}  |
   +--------------+---------------+    +--------+-------+    +--+---+
                  |                             |               |
                  |          +----------------- +               |
                  +---+      |                                  |
                      |      |      +---------------------------+
                      |      |      |
                      |      |      |
                      v      v      v        +----------------------+    +----------------------+
                  +---+------+------+---+    |          H           |    |          C           |
                  |        Gather       |    | (merged with H_out)  |    | (merged with C_out)  |    +-----+    +-----+    +-----+
                  | [batch, input_size] |    | [batch, hidden_size] |    | [batch, hidden_size] |    |  W  |    |  R  |    |  B  |
                  +----------+----------+    +----------+-----------+    +----------+-----------+    +--+--+    +--+--+    +--+--+
                             |                          |                           |                   |          |          |
                             |                          |                           |                   |          |          |
                             |                          |                           |                   |          |          |
                             |                          |                           |                   |          |          |
                             |                          |                           |                   |          |          |
                             |                          |                           |                   |          |          |
                             |                          |           +---------------+                   |          |          |
                             |                          |           |                                   |          |          |
                             |                          |           |                                   |          |          |
                             |                          |           |    +------------------------------+          |          |
                             |                          |           |    |                                         |          |
                             |                          |           |    |                                         |          |
                             |                          +------+    |    |    +------------------------------------+          |
                             |                                 |    |    |    |                                               |
                             +----------------------------+    |    |    |    |    +------------------------------------------+
                                                          |    |    |    |    |    |
       +---+                                              v    v    v    v    v    v
       | Y |                                          +---+----+----+----+----+----+---+
       +---+                                          |            LSTMCell            |
         |                                            +--------+-------------------+---+
         |                                                     |                   |
         v                                                     |                   |
   +-----+-----+                                    +----------+---------------+   |
   | Broadcast |                                    |                          |   +---------------------+
   +-----+-----+                                    |                          |                         |
         |                                          v                          v                         v
         |       +----------------+    +------------+------------+   +---------+------------+   +--------+--------+
         |       | sequence index |    |       Unsqueeze         |   |         H_out        |   |      C_out      |
         |       +--------+-------+    | [batch, 1, hidden_size] |   |   (merged with H)    |   | (merged with C) |
         |                |            +------------+------------+   | [batch, hidden_size] |   +-----------------+
         |                |                         |                +----------------------+
         |                |                         |
         |                |                         |
         |                |                         |      +------+
         |                |                         |      | axis |
         |                |                         |      | {0}  |
         |                |        +----------------+      +--+---+
         |                |        |                          |
         |                |        |                          |
         |                +---+    |    +---------------------+
         |                    |    |    |
         |                    |    |    |
         +---------------+    |    |    |
                         |    |    |    |
                         v    v    v    v
                     +---+----+----+----+---+
                     |     ScatterUpdate    |
                     |  (loop body output)  |
                     +----------------------+

*/
// clang-format on

ov::pass::ConvertLoopWithScatterUpdateToLSTMSequence::ConvertLoopWithScatterUpdateToLSTMSequence() {
    MATCHER_SCOPE(ConvertLoopWithScatterUpdateToLSTMSequence);
    auto input_label = pattern::any_input(pattern::rank_equals(3));
    auto input_transpose_const_label = pattern::wrap_type<op::v0::Constant>();
    auto input_transpose_label =
        pattern::wrap_type<op::v1::Transpose, op::v1::Reshape>({input_label, input_transpose_const_label},
                                                               pattern::rank_equals(3));
    auto scatter_indexes_label = pattern::wrap_type<op::v0::Constant>();
    auto scatter_update_label = std::make_shared<pattern::op::Or>(OutputVector{input_transpose_label, input_label});
    auto scatter_label = pattern::wrap_type<op::v3::ScatterNDUpdate>(
        {pattern::any_input(), scatter_indexes_label, scatter_update_label});
    auto trip_count_label = pattern::wrap_type<op::v0::Constant>();
    auto cond_label = pattern::wrap_type<op::v0::Constant>();
    auto loop_label = pattern::wrap_type<op::v5::Loop>({trip_count_label,
                                                        cond_label,
                                                        pattern::any_input(),
                                                        pattern::any_input(),
                                                        pattern::any_input(),
                                                        pattern::any_input(),
                                                        pattern::any_input(),
                                                        scatter_label});
    auto output_transpose_const_label = pattern::wrap_type<op::v0::Constant>();
    auto output_transpose_label = pattern::wrap_type<op::v1::Transpose>({loop_label, output_transpose_const_label});

    // Loop body pattern:
    auto sequence_index_label = pattern::any_input(pattern::rank_equals(0));
    auto iteration_counter_label = pattern::any_input();
    auto iteration_counter_step_label = pattern::wrap_type<op::v0::Constant>();
    auto iteration_counter_incremented_label =
        pattern::wrap_type<op::v1::Add>({iteration_counter_label, iteration_counter_step_label});
    auto iteration_counter_limit_label = pattern::wrap_type<op::v0::Constant>();
    auto iteration_counter_less_than_limit_label =
        pattern::wrap_type<op::v1::Less>({iteration_counter_incremented_label, iteration_counter_limit_label});
    auto sequence_index_step_label = pattern::wrap_type<op::v0::Constant>();
    auto sequence_index_incremented_label =
        pattern::wrap_type<op::v1::Add>({sequence_index_label, sequence_index_step_label});
    auto sequence_index_limit_label = pattern::wrap_type<op::v0::Constant>();
    auto sequence_index_less_than_limit_label =
        pattern::wrap_type<op::v1::Less>({sequence_index_incremented_label, sequence_index_limit_label});
    auto and_label = pattern::wrap_type<op::v1::LogicalAnd>(
        {iteration_counter_less_than_limit_label, sequence_index_less_than_limit_label});
    auto loop_condition_label = pattern::wrap_type<op::v0::Result>({and_label});

    auto X_body_label = pattern::any_input(pattern::rank_equals(3));
    auto C_body_label = pattern::any_input(pattern::rank_equals(2));
    auto H_body_label = pattern::any_input(pattern::rank_equals(2));
    auto gather_axis_label = pattern::wrap_type<op::v0::Constant>();
    auto sequence_index_new_shape_label = pattern::wrap_type<op::v0::Constant>();
    auto sequence_index_reshaped_label =
        pattern::wrap_type<op::v1::Reshape>({sequence_index_label, sequence_index_new_shape_label});
    auto sequence_index_or_label =
        std::make_shared<pattern::op::Or>(OutputVector{sequence_index_label, sequence_index_reshaped_label});
    auto gather_body_label =
        pattern::wrap_type<ov::op::v8::Gather>({X_body_label, sequence_index_or_label, gather_axis_label},
                                               pattern::rank_equals(2));
    auto W_label = pattern::any_input();
    auto R_label = pattern::any_input();
    auto B_label = pattern::wrap_type<op::v0::Constant>();
    auto lstm_cell_label = pattern::wrap_type<ov::op::v4::LSTMCell>(
        {gather_body_label, H_body_label, C_body_label, W_label, R_label, B_label});
    auto scatter_index_new_shape_label = pattern::wrap_type<op::v0::Constant>();
    auto scatter_index_body_label =
        pattern::wrap_type<op::v1::Reshape>({sequence_index_label, scatter_index_new_shape_label});
    auto updates_label = pattern::wrap_type<op::v1::Reshape, op::v0::Unsqueeze>(
        {lstm_cell_label, pattern::wrap_type<op::v0::Constant>()});
    auto scatter_axis_label = pattern::wrap_type<op::v0::Constant>();
    auto scatter_body_label = pattern::wrap_type<op::v3::ScatterUpdate>(
        {pattern::any_input(), scatter_index_body_label, updates_label, scatter_axis_label},
        pattern::rank_equals(3));
    auto loop_output_label = pattern::wrap_type<op::v0::Result>({scatter_body_label});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto match_root = m.get_match_root();

        const auto loop = ov::as_type_ptr<op::v5::Loop>(pattern_map.at(loop_label).get_node_shared_ptr());
        const auto& output_descs = loop->get_output_descriptions();
        if (output_descs.size() != 1)
            return false;
        const auto body_output_desc =
            std::dynamic_pointer_cast<op::util::MultiSubGraphOp::BodyOutputDescription>(output_descs[0]);
        if (!body_output_desc || body_output_desc->m_iteration != -1)
            return false;

        ov::pass::pattern::Matcher loop_condition_matcher(loop_condition_label);
        ov::pass::pattern::Matcher loop_output_matcher(loop_output_label);

        auto body = loop->get_function();
        const auto& body_parameters = body->get_parameters();
        const auto& body_results = body->get_results();
        const auto special_body_ports = loop->get_special_body_ports();

        if (!loop_condition_matcher.match(body_results[special_body_ports.body_condition_output_idx]->output(0)))
            return false;
        if (!loop_output_matcher.match(body_results[body_output_desc->m_body_value_index]->output(0)))
            return false;

        const auto& loop_condition_map = loop_condition_matcher.get_pattern_value_map();
        const auto& loop_output_map = loop_output_matcher.get_pattern_value_map();

        int64_t iteration_counter_step = -1;
        if (!get_constant_value(loop_condition_map.at(iteration_counter_step_label).get_node_shared_ptr(),
                                iteration_counter_step) ||
            iteration_counter_step != 1)
            return false;
        int64_t sequence_index_step = -1;
        if (!get_constant_value(loop_condition_map.at(sequence_index_step_label).get_node_shared_ptr(),
                                sequence_index_step) ||
            sequence_index_step != 1)
            return false;

        int64_t iteration_counter_limit = -1;
        if (!get_constant_value(loop_condition_map.at(iteration_counter_limit_label).get_node_shared_ptr(),
                                iteration_counter_limit))
            return false;
        int64_t sequence_index_limit = -1;
        if (!get_constant_value(loop_condition_map.at(sequence_index_limit_label).get_node_shared_ptr(),
                                sequence_index_limit))
            return false;
        if (iteration_counter_limit != sequence_index_limit)
            return false;

        int64_t gather_axis = -1;
        if (!get_constant_value(loop_output_map.at(gather_axis_label).get_node_shared_ptr(), gather_axis) ||
            gather_axis != 0)
            return false;
        int64_t scatter_axis = -1;
        if (!get_constant_value(loop_output_map.at(scatter_axis_label).get_node_shared_ptr(), scatter_axis) ||
            scatter_axis != 0)
            return false;

        const auto& sequence_index = loop_condition_map.at(sequence_index_label).get_node_shared_ptr();
        const auto& iteration_counter = loop_condition_map.at(iteration_counter_label).get_node_shared_ptr();

        const auto& X_body = loop_output_map.at(X_body_label).get_node_shared_ptr();
        const auto& H_body = loop_output_map.at(H_body_label).get_node_shared_ptr();
        const auto& C_body = loop_output_map.at(C_body_label).get_node_shared_ptr();
        auto W = loop_output_map.at(W_label).get_node_shared_ptr();
        auto R = loop_output_map.at(R_label).get_node_shared_ptr();
        auto B = loop_output_map.at(B_label).get_node_shared_ptr();
        const auto lstm_cell =
            ov::as_type_ptr<op::v4::LSTMCell>(loop_output_map.at(lstm_cell_label).get_node_shared_ptr());
        const auto H_unsqueeze = loop_output_map.at(updates_label).get_node_shared_ptr();
        if (H_unsqueeze->input_value(0) != lstm_cell->output(0))
            return false;

        Output<Node> X = pattern_map.at(input_label);
        Output<Node> H;
        Output<Node> C;

        const auto& input_descs = loop->get_input_descriptions();
        for (const auto& desc : input_descs) {
            if (body_parameters[desc->m_body_parameter_index] == X_body) {
                if (!std::dynamic_pointer_cast<op::util::MultiSubGraphOp::InvariantInputDescription>(desc)) {
                    return false;
                }
                if (loop->input_value(desc->m_input_index) != pattern_map.at(scatter_label)) {
                    return false;
                }
            }
            if (body_parameters[desc->m_body_parameter_index] == H_body) {
                auto merged_desc = std::dynamic_pointer_cast<op::util::MultiSubGraphOp::MergedInputDescription>(desc);
                if (!merged_desc) {
                    return false;
                }
                H = loop->input_value(desc->m_input_index);
                const auto& result = body_results[merged_desc->m_body_value_index];
                if (result->input_value(0) != lstm_cell->output(0)) {
                    return false;
                }
            }
            if (body_parameters[desc->m_body_parameter_index] == C_body) {
                auto merged_desc = std::dynamic_pointer_cast<op::util::MultiSubGraphOp::MergedInputDescription>(desc);
                if (!merged_desc) {
                    return false;
                }
                C = loop->input_value(desc->m_input_index);
                const auto& result = body_results[merged_desc->m_body_value_index];
                if (result->input_value(0) != lstm_cell->output(1)) {
                    return false;
                }
            }
            if (body_parameters[desc->m_body_parameter_index] == sequence_index) {
                auto merged_desc = std::dynamic_pointer_cast<op::util::MultiSubGraphOp::MergedInputDescription>(desc);
                if (!merged_desc) {
                    return false;
                }
            }
            if (body_parameters[desc->m_body_parameter_index] == iteration_counter) {
                auto merged_desc = std::dynamic_pointer_cast<op::util::MultiSubGraphOp::MergedInputDescription>(desc);
                if (!merged_desc) {
                    return false;
                }
            }
        }

        auto constant_is_zero = [](const Output<Node>& node) -> bool {
            auto constant = ov::as_type_ptr<op::v0::Constant>(node.get_node_shared_ptr());
            if (!constant) {
                return false;
            }
            float value = -1.0f;
            return ov::op::util::get_single_value(constant, value) && value == 0.0f;
        };

        if (!constant_is_zero(H))
            return false;
        if (!constant_is_zero(C))
            return false;

        const auto& scatter = pattern_map.at(scatter_label);
        const auto& scatter_shape = scatter.get_partial_shape();  // scatter shape [sequence length, batch, input size]
        const auto& sequence_length_dimension = scatter_shape[0];
        const auto& batch_size_dimension = scatter_shape[1];
        const auto& input_size_dimension = scatter_shape[2];

        std::vector<int> batch_first_perm{1, 0, 2};
        std::vector<int> new_input_perm_values;

        if (pattern_map.count(input_transpose_label) > 0) {
            const auto& input_transpose = pattern_map.at(input_transpose_label).get_node();
            if (ov::is_type<op::v1::Transpose>(input_transpose)) {
                auto input_perm = ov::as_type<op::v0::Constant>(input_transpose->get_input_node_ptr(1));
                if (!input_perm)
                    return false;
                auto input_perm_values = input_perm->cast_vector<int>();
                for (size_t i = 0; i < input_perm_values.size(); i++) {
                    new_input_perm_values.push_back(input_perm_values[batch_first_perm[i]]);
                }
            } else if (ov::is_type<op::v1::Reshape>(input_transpose)) {
                const auto& input_shape = input_transpose->get_input_partial_shape(0);
                const auto& output_shape = input_transpose->get_output_partial_shape(0);
                if (input_shape.size() != output_shape.size())
                    return false;
                for (size_t i = 0; i < output_shape.size(); i++) {
                    const auto& dim = output_shape[i];
                    for (size_t j = 0; j < input_shape.size(); j++) {
                        if (input_shape[j] == dim) {
                            new_input_perm_values.push_back(batch_first_perm[j]);
                            break;
                        }
                    }
                }
            }
        } else {
            new_input_perm_values = batch_first_perm;
        }

        NodeRegistry node_registry;

        if (new_input_perm_values != std::vector<int>{0, 1, 2}) {
            auto new_input_perm = node_registry.make<op::v0::Constant>(element::i32,
                                                                       Shape{new_input_perm_values.size()},
                                                                       new_input_perm_values);
            X = node_registry.make<op::v1::Transpose>(X, new_input_perm);
        }

        const auto& X_shape = X.get_partial_shape();
        if (!X_shape[0].compatible(batch_size_dimension) || !X_shape[1].compatible(sequence_length_dimension) ||
            !X_shape[2].compatible(input_size_dimension)) {
            return false;
        }

        // Finally create LSTMSequence
        auto zero = node_registry.make<op::v0::Constant>(element::i32, Shape{1}, 0);
        auto max_sequence_length = node_registry.make<op::v0::Constant>(element::i32, Shape{1}, sequence_index_limit);
        auto shapeof_X = node_registry.make<op::v3::ShapeOf>(X);
        auto batch_size = node_registry.make<op::v8::Gather>(shapeof_X, zero, zero);
        auto shapeof_H = node_registry.make<op::v3::ShapeOf>(H);
        auto new_H_shape = node_registry.make<op::v0::Concat>(OutputVector{batch_size, shapeof_H}, 0);
        auto new_H = node_registry.make<op::v3::Broadcast>(H, new_H_shape);
        auto shapeof_C = node_registry.make<op::v3::ShapeOf>(C);
        auto new_C_shape = node_registry.make<op::v0::Concat>(OutputVector{batch_size, shapeof_C}, 0);
        auto new_C = node_registry.make<op::v3::Broadcast>(C, new_C_shape);
        auto new_W = node_registry.make<op::v0::Unsqueeze>(W, zero);
        auto new_R = node_registry.make<op::v0::Unsqueeze>(R, zero);
        auto new_B = node_registry.make<op::v0::Unsqueeze>(B, zero);
        std::shared_ptr<Node> sequence_lengths = std::make_shared<op::v3::Broadcast>(max_sequence_length, batch_size);
        if (auto constant = ov::util::constantfold_subgraph(sequence_lengths)) {
            sequence_lengths = constant;
        }
        node_registry.add(sequence_lengths);
        auto lstm = node_registry.make<op::v5::LSTMSequence>(X,
                                                             new_H,
                                                             new_C,
                                                             sequence_lengths,
                                                             new_W,
                                                             new_R,
                                                             new_B,
                                                             lstm_cell->get_hidden_size(),
                                                             op::v5::LSTMSequence::direction::FORWARD,
                                                             lstm_cell->get_activations_alpha(),
                                                             lstm_cell->get_activations_beta(),
                                                             lstm_cell->get_activations(),
                                                             lstm_cell->get_clip());
        if (transformation_callback(lstm))
            return false;

        const auto one = node_registry.make<op::v0::Constant>(element::i32, Shape{1}, 1);
        auto H_squeezed = node_registry.make<op::v0::Squeeze>(lstm->output(0), one);
        H_squeezed->set_friendly_name(match_root->get_friendly_name());

        copy_runtime_info(NodeVector{scatter.get_node_shared_ptr(), loop}, node_registry.get());

        for (auto&& loop_consumer : loop->output(0).get_target_inputs()) {
            auto node = loop_consumer.get_node()->shared_from_this();
            if (ov::is_type<op::util::ShapeOfBase>(node)) {
                auto shapeof = std::make_shared<op::v3::ShapeOf>(H_squeezed);
                auto indices = op::v0::Constant::create(element::i32, Shape{3}, {1, 0, 2});
                auto shapeof_gather = std::make_shared<op::v8::Gather>(shapeof, indices, zero);
                shapeof_gather->set_friendly_name(node->get_friendly_name());
                copy_runtime_info(node, {shapeof, indices, shapeof_gather});
                replace_node(node, shapeof_gather);
            }
        }

        replace_node(match_root, H_squeezed);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(output_transpose_label, matcher_name);
    register_matcher(m, callback);
}

// clang-format off
/*

   Following subgraph in Loop is fused into LSTMSequence

   +------------------------------+
   |              X               |                          +------+
   |         (sliced input)       |                          | axis |
   |     [1, batch, input_size]   |                          | {0}  |
   +--------------+---------------+                          +--+---+
                  |                                             |
                  |                                             |
                  +---+                                         |
                      |             +---------------------------+
                      |             |
                      |             |
                      v             v        +----------------------+    +----------------------+
                  +---+------+------+---+    |          H           |    |          C           |
                  |        Squeeze      |    | (merged with H_out)  |    | (merged with C_out)  |    +-----+    +-----+    +-----+
                  | [batch, input_size] |    | [batch, hidden_size] |    | [batch, hidden_size] |    |  W  |    |  R  |    |  B  |
                  +----------+----------+    +----------+-----------+    +----------+-----------+    +--+--+    +--+--+    +--+--+
                             |                          |                           |                   |          |          |
                             |                          |           +---------------+                   |          |          |
                             |                          |           |                                   |          |          |
                             |                          |           |    +------------------------------+          |          |
                             |                          |           |    |                                         |          |
                             |                          +------+    |    |    +------------------------------------+          |
                             +----------------------------+    |    |    |    |    +------------------------------------------+
                                                          |    |    |    |    |    |
                                                          v    v    v    v    v    v
                                                      +---+----+----+----+----+----+---+
                                                      |            LSTMCell            |
                                                      +--------+-------------------+---+
                                                               |                   |
                                                               |                   |
                                                    +----------+---------------+   |
                                                    |                          |   +---------------------+
                                                    |                          |                         |
                                                    v                          v                         v
                                       +------------+------------+   +---------+------------+   +--------+--------+
                                       |   Unsqueeze, axis=0     |   |         H_out        |   |      C_out      |
                                       | [batch, 1, hidden_size] |   |   (merged with H)    |   | (merged with C) |
                                       |   (concat output)       |   | [batch, hidden_size] |   |                 |
                                       +-------------------------+   +----------------------+   +-----------------+

*/
// clang-format on

ov::pass::ConvertLoopWithSlicedInputConcatOutputToLSTMSequence::ConvertLoopWithSlicedInputConcatOutputToLSTMSequence() {
    MATCHER_SCOPE(ConvertLoopWithSlicedInputConcatOutputToLSTMSequence);

    auto loop_label = pattern::wrap_type<ov::op::v5::Loop>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto loop = ov::as_type_ptr<ov::op::v5::Loop>(m.get_match_root());
        if (!loop) {
            return false;
        }

        const auto& body_graph = loop->get_function();
        const auto& body_params = body_graph->get_parameters();
        const auto& body_results = body_graph->get_results();
        const auto& special_body_ports = loop->get_special_body_ports();
        const auto& input_descriptions = loop->get_input_descriptions();
        const auto& output_descriptions = loop->get_output_descriptions();
        if (special_body_ports.body_condition_output_idx < 0) {
            return false;
        }

        // check condition pattern and retrieve a number of iterations
        ov::Output<ov::Node> num_iters_output;
        if (!check_condition_increment_pattern(body_results[special_body_ports.body_condition_output_idx],
                                               loop,
                                               body_params,
                                               body_results,
                                               input_descriptions,
                                               num_iters_output) &&
            !check_condition_true_pattern(body_results[special_body_ports.body_condition_output_idx],
                                          loop,
                                          num_iters_output)) {
            return false;
        }

        // check body pattern with LSTMCell
        // and extract key points for LSTMSequence creation
        ov::Output<Node> init_hidden_state, init_cell_state, x, W, R, B;
        std::shared_ptr<op::v0::Result> result_hidden_state;
        std::shared_ptr<ov::op::v4::LSTMCell> lstm_cell_node;
        if (!check_lstm_cell_pattern(loop,
                                     body_params,
                                     body_results,
                                     input_descriptions,
                                     init_hidden_state,
                                     init_cell_state,
                                     x,
                                     W,
                                     R,
                                     B,
                                     result_hidden_state,
                                     lstm_cell_node)) {
            return false;
        }

        // check that only output with concatenated hidden states from Loop is needed
        // or get hidden state from last iteration
        std::vector<uint64_t> last_iter_hidden_state_output_ids;
        std::vector<uint64_t> all_hidden_states_output_ids;
        for (const auto& output_desc : output_descriptions) {
            auto result_idx = output_desc->m_body_value_index;
            auto output_idx = output_desc->m_output_index;
            if (body_results[result_idx] != result_hidden_state) {
                if (loop->get_output_target_inputs(output_idx).size() > 0) {
                    // some other Result node value from body graph is required by real consumers
                    return false;
                }
                continue;
            }
            auto concat_desc = ov::as_type_ptr<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>(output_desc);
            if (concat_desc) {
                // all hidden states are concatenated
                if ((concat_desc->m_axis != 0) || (concat_desc->m_start != 0) || (concat_desc->m_end != -1) ||
                    (concat_desc->m_stride != 1)) {
                    return false;
                }
                all_hidden_states_output_ids.push_back(output_idx);
                continue;
            }
            last_iter_hidden_state_output_ids.push_back(output_idx);
        }

        // all input and output ports are almost ready
        // preparation of the right format of inputs and outputs is remained before reconnection
        NodeRegistry rg;
        // x is in a format [seq_len, batch_size, input_size]
        auto tr_order = rg.make<ov::op::v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{1, 0, 2});
        auto x_prep = rg.make<ov::op::v1::Transpose>(x, tr_order);
        // initial hidden and cell states are of shape [batch_size, hidden_size]
        // needs to add num_directions size
        auto unsqueeze_axis = rg.make<ov::op::v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{1});
        auto init_hidden_state_prep = rg.make<ov::op::v0::Unsqueeze>(init_hidden_state, unsqueeze_axis);
        auto init_cell_state_prep = rg.make<ov::op::v0::Unsqueeze>(init_cell_state, unsqueeze_axis);
        // sequence length is currently scalar and it needs broadcasting to shape [batch_size]
        auto x_shape = rg.make<ov::op::v3::ShapeOf>(x, element::i64);
        auto ss_start = rg.make<ov::op::v0::Constant>(element::i64, Shape{1}, 1);
        auto ss_stop = rg.make<ov::op::v0::Constant>(element::i64, Shape{1}, 2);
        auto ss_step = rg.make<ov::op::v0::Constant>(element::i64, Shape{1}, 1);
        auto batch_size = rg.make<ov::op::v1::StridedSlice>(x_shape,
                                                            ss_start,
                                                            ss_stop,
                                                            ss_step,
                                                            std::vector<int64_t>{0},
                                                            std::vector<int64_t>{0});
        auto seq_lens_prep = rg.make<ov::op::v1::Broadcast>(num_iters_output, batch_size);
        // prepare W, R, B to add num_directions dimension
        unsqueeze_axis = rg.make<ov::op::v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
        auto W_prep = rg.make<ov::op::v0::Unsqueeze>(W, unsqueeze_axis);
        auto R_prep = rg.make<ov::op::v0::Unsqueeze>(R, unsqueeze_axis);
        auto B_prep = rg.make<ov::op::v0::Unsqueeze>(B, unsqueeze_axis);

        // create LSTMSequence operation since all inputs are prepared
        // extract hidden size that should be static as a requirement of OpenVINO LSTMCell operation
        int64_t hidden_size = static_cast<int64_t>(lstm_cell_node->get_hidden_size());
        auto lstm_sequence = rg.make<ov::op::v5::LSTMSequence>(x_prep,
                                                               init_hidden_state_prep,
                                                               init_cell_state_prep,
                                                               seq_lens_prep,
                                                               W_prep,
                                                               R_prep,
                                                               B_prep,
                                                               hidden_size,
                                                               ov::op::RecurrentSequenceDirection::FORWARD,
                                                               lstm_cell_node->get_activations_alpha(),
                                                               lstm_cell_node->get_activations_beta(),
                                                               lstm_cell_node->get_activations(),
                                                               lstm_cell_node->get_clip());

        if (transformation_callback(lstm_sequence))
            return false;

        // prepare outputs of LSTMSequence
        // output with concatenated hidden states must be in a format [seq_len, batch_size, hidden_size]
        // LSTMSequence generates it in a format [batch_size, num_directions, seq_len, hidden_size]
        auto squeeze_axis = rg.make<ov::op::v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{1});
        ov::Output<Node> all_hidden_states_prep = rg.make<ov::op::v0::Squeeze>(lstm_sequence->output(0), squeeze_axis);
        all_hidden_states_prep = rg.make<ov::op::v1::Transpose>(all_hidden_states_prep, tr_order);
        // prepare output with last hidden state
        // LSTMSequence operation outputs it in a format [batch_size, num_directions, hidden_size]
        auto tr_order2 = rg.make<ov::op::v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{1, 0, 2});
        ov::Output<ov::Node> last_hidden_state_prep =
            rg.make<ov::op::v1::Transpose>(lstm_sequence->output(1), tr_order2);

        // reconnect all consumers for hidden states
        for (auto output_idx : all_hidden_states_output_ids) {
            loop->output(output_idx).replace(all_hidden_states_prep);
        }
        for (auto output_idx : last_iter_hidden_state_output_ids) {
            loop->output(output_idx).replace(last_hidden_state_prep);
        }

        copy_runtime_info(m.get_matched_nodes(), rg.get());
        if ((all_hidden_states_output_ids.size() > 0) && (last_iter_hidden_state_output_ids.size() == 0)) {
            all_hidden_states_prep.get_node_shared_ptr()->set_friendly_name(loop->get_friendly_name());
        } else if ((all_hidden_states_output_ids.size() == 0) && (last_iter_hidden_state_output_ids.size() > 0)) {
            last_hidden_state_prep.get_node_shared_ptr()->set_friendly_name(loop->get_friendly_name());
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(loop_label, matcher_name);
    register_matcher(m, callback);
}

class EliminateGatherWithRange : public ov::pass::MatcherPass {
public:
    EliminateGatherWithRange() {
        using namespace ov;
        using namespace ov::pass;

        auto data_label = pattern::any_input(pattern::rank_equals(3));
        auto shapeof_label = pattern::wrap_type<op::util::ShapeOfBase>({data_label});
        auto shapeof_gather_label = pattern::wrap_type<op::util::GatherBase>(
            {shapeof_label, pattern::wrap_type<op::v0::Constant>(), pattern::wrap_type<op::v0::Constant>()});
        auto shapeof_gather2_label = pattern::wrap_type<op::util::GatherBase>(
            {shapeof_gather_label, pattern::wrap_type<op::v0::Constant>(), pattern::wrap_type<op::v0::Constant>()});
        auto reshape_label =
            pattern::wrap_type<op::v1::Reshape>({shapeof_gather2_label, pattern::wrap_type<op::v0::Constant>()});
        auto range_label = pattern::wrap_type<op::v4::Range>(
            {pattern::wrap_type<op::v0::Constant>(), reshape_label, pattern::wrap_type<op::v0::Constant>()});
        auto match_node = pass::pattern::wrap_type<op::util::GatherBase>(
            {data_label, range_label, pattern::wrap_type<op::v0::Constant>()});

        matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto gather = ov::as_type_ptr<op::util::GatherBase>(m.get_match_root());
            if (!gather)
                return false;
            auto axis = gather->get_axis();
            if (axis == op::v1::Gather::AXIS_NOT_SET_VALUE) {
                return false;
            }

            const auto shapeof_gather = pattern_map.at(shapeof_gather_label).get_node_shared_ptr();
            const auto shapeof_gather_indexes_node =
                ov::as_type_ptr<op::v0::Constant>(shapeof_gather->get_input_node_shared_ptr(1));
            auto shapeof_gather_indexes = shapeof_gather_indexes_node->cast_vector<int64_t>();
            if (shapeof_gather_indexes.size() != 3)
                return false;
            const auto shapeof_gather2 = pattern_map.at(shapeof_gather2_label).get_node_shared_ptr();
            int64_t shapeof_gather2_index = -1;
            int64_t shapeof_gather2_axis = -1;
            if (!get_constant_value(shapeof_gather2->get_input_node_shared_ptr(1), shapeof_gather2_index))
                return false;
            if (!get_constant_value(shapeof_gather2->get_input_node_shared_ptr(2), shapeof_gather2_axis) ||
                shapeof_gather2_axis != 0)
                return false;
            const auto reshape = pattern_map.at(reshape_label).get_node_shared_ptr();
            const auto& reshape_shape = reshape->get_output_partial_shape(0);
            if (reshape_shape.is_dynamic() || reshape_shape.size() != 0)
                return false;
            const auto range = pattern_map.at(range_label).get_node_shared_ptr();
            int64_t range_start = -1;
            int64_t range_step = -1;
            if (!get_constant_value(range->get_input_node_shared_ptr(0), range_start) || range_start != 0)
                return false;
            if (!get_constant_value(range->get_input_node_shared_ptr(2), range_step) || range_step != 1)
                return false;

            int64_t gather_axis = -1;
            if (!get_constant_value(gather->get_input_node_shared_ptr(2), gather_axis) ||
                gather_axis != shapeof_gather_indexes[shapeof_gather2_index])
                return false;

            return replace_output_update_name(gather->output(0), gather->input_value(0));
        };

        auto m = std::make_shared<pattern::Matcher>(match_node, "EliminateGatherWithRange");
        register_matcher(m, callback);
    }
};

ov::pass::FuseReverseLSTMSequence::FuseReverseLSTMSequence() {
    MATCHER_SCOPE(FuseReverseLSTMSequence);

    auto data_label = pattern::any_input(pattern::rank_equals(3));
    auto first_transpose_label =
        pattern::wrap_type<op::v1::Transpose, op::v1::Reshape>({data_label, pattern::wrap_type<op::v0::Constant>()},
                                                               pattern::rank_equals(3));
    auto input_to_first_reverse_sequence_label =
        std::make_shared<pattern::op::Or>(OutputVector{first_transpose_label, data_label});
    auto first_reverse_sequence_label =
        pattern::wrap_type<op::v0::ReverseSequence>({input_to_first_reverse_sequence_label, pattern::any_input()});
    auto second_transpose_label =
        pattern::wrap_type<op::v1::Transpose>({first_reverse_sequence_label, pattern::wrap_type<op::v0::Constant>()});
    auto lstm_label = pattern::wrap_type<op::v5::LSTMSequence>({second_transpose_label,
                                                                pattern::any_input(),
                                                                pattern::any_input(),
                                                                pattern::any_input(),
                                                                pattern::any_input(),
                                                                pattern::any_input(),
                                                                pattern::any_input()},
                                                               pattern::consumers_count(1));
    auto squeeze_label = pattern::wrap_type<op::v0::Squeeze>({lstm_label, pattern::wrap_type<op::v0::Constant>()});
    auto second_reverse_sequence_label =
        pattern::wrap_type<op::v0::ReverseSequence>({squeeze_label, pattern::any_input()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& data = pattern_map.at(data_label);
        const auto second_transpose = pattern_map.at(second_transpose_label).get_node_shared_ptr();
        const auto second_transpose_perm =
            ov::as_type_ptr<op::v0::Constant>(second_transpose->get_input_node_shared_ptr(1));
        auto lstm = ov::as_type_ptr<op::v5::LSTMSequence>(pattern_map.at(lstm_label).get_node_shared_ptr());
        if (lstm->get_direction() != op::v5::LSTMSequence::direction::FORWARD)
            return false;

        std::shared_ptr<Node> new_transpose_perm;
        if (pattern_map.count(first_transpose_label) > 0) {
            auto first_transpose = pattern_map.at(first_transpose_label).get_node_shared_ptr();
            if (ov::is_type<op::v1::Reshape>(first_transpose)) {
                const auto& reshape_input_shape = first_transpose->get_input_partial_shape(0);
                const auto& reshape_output_shape = first_transpose->get_output_partial_shape(0);
                if (reshape_input_shape.size() != reshape_output_shape.size())
                    return false;
                const auto second_transpose_perm_values = second_transpose_perm->cast_vector<int>();
                std::vector<int> new_perm_values;
                for (size_t i = 0; i < reshape_output_shape.size(); i++) {
                    const auto& dim = reshape_output_shape[i];
                    for (size_t j = 0; j < reshape_input_shape.size(); j++) {
                        if (dim == reshape_input_shape[j]) {
                            new_perm_values.push_back(second_transpose_perm_values[j]);
                        }
                    }
                }
                if (new_perm_values.size() != 3)
                    return false;
                if (new_perm_values != std::vector<int>{0, 1, 2}) {
                    new_transpose_perm =
                        op::v0::Constant::create(element::i32, Shape{new_perm_values.size()}, new_perm_values);
                }
            } else if (ov::is_type<op::v1::Transpose>(first_transpose)) {
                const auto first_transpose_perm = ov::as_type<op::v0::Constant>(first_transpose->get_input_node_ptr(1));
                const auto first_transpose_perm_values = first_transpose_perm->cast_vector<int>();
                const auto second_transpose_perm_values = second_transpose_perm->cast_vector<int>();
                if (first_transpose_perm_values.size() != second_transpose_perm_values.size())
                    return false;
                std::vector<int> new_perm_values;
                for (size_t i = 0; i < first_transpose_perm_values.size(); i++) {
                    new_perm_values.push_back(first_transpose_perm_values[second_transpose_perm_values[i]]);
                }
                if (new_perm_values.size() != 3)
                    return false;
                if (new_perm_values != std::vector<int>{0, 1, 2}) {
                    new_transpose_perm =
                        op::v0::Constant::create(element::i32, Shape{new_perm_values.size()}, new_perm_values);
                }
            }
        } else {
            new_transpose_perm = second_transpose_perm;
        }

        NodeRegistry node_registry;

        Output<Node> new_lstm_input;
        if (new_transpose_perm) {
            new_lstm_input = node_registry.make<op::v1::Transpose>(data, new_transpose_perm);
        } else {
            new_lstm_input = data;
        }

        auto new_lstm = node_registry.make<op::v5::LSTMSequence>(new_lstm_input,
                                                                 lstm->input_value(1),
                                                                 lstm->input_value(2),
                                                                 lstm->input_value(3),
                                                                 lstm->input_value(4),
                                                                 lstm->input_value(5),
                                                                 lstm->input_value(6),
                                                                 lstm->get_hidden_size(),
                                                                 op::v5::LSTMSequence::direction::REVERSE,
                                                                 lstm->get_activations_alpha(),
                                                                 lstm->get_activations_beta(),
                                                                 lstm->get_activations(),
                                                                 lstm->get_clip());

        auto squeeze = pattern_map.at(squeeze_label).get_node_shared_ptr();
        if (squeeze->input_value(0) != lstm->output(0))
            return false;
        int64_t squeeze_axis = -1;
        if (!get_constant_value(squeeze->get_input_node_shared_ptr(1), squeeze_axis) || squeeze_axis != 1)
            return false;
        auto new_squeeze = node_registry.make<op::v0::Squeeze>(new_lstm->output(0), squeeze->input_value(1));
        const auto match_root = m.get_match_root();
        new_squeeze->set_friendly_name(match_root->get_friendly_name());

        for (auto& consumer : second_transpose->output(0).get_target_inputs()) {
            auto node = consumer.get_node()->shared_from_this();
            if (ov::is_type<op::util::ShapeOfBase>(node)) {
                auto shapeof = std::make_shared<op::v3::ShapeOf>(new_lstm_input);
                replace_node(node, shapeof);
            }
        }

        NodeVector from{pattern_map.at(first_reverse_sequence_label).get_node_shared_ptr(),
                        second_transpose,
                        lstm,
                        squeeze,
                        pattern_map.at(second_reverse_sequence_label).get_node_shared_ptr()};
        if (pattern_map.count(first_transpose_label) > 0) {
            from.push_back(pattern_map.at(first_transpose_label).get_node_shared_ptr());
        }

        copy_runtime_info(from, node_registry.get());
        replace_node(match_root, new_squeeze);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(second_reverse_sequence_label, matcher_name);
    register_matcher(m, callback);
}

ov::pass::FuseLSTMSequencesToBidirectionalLSTMSequence::FuseLSTMSequencesToBidirectionalLSTMSequence() {
    MATCHER_SCOPE(FuseLSTMSequencesToBidirectionalLSTMSequence);
    auto data_label = pattern::any_input();

    // forward pattern
    auto transpose_forward_label =
        pattern::wrap_type<op::v1::Transpose>({data_label, pattern::wrap_type<op::v0::Constant>()});
    auto lstm_sequence_forward_first_input_label =
        std::make_shared<pattern::op::Or>(OutputVector{transpose_forward_label, data_label});
    auto shapeof_forward_label = pattern::wrap_type<op::util::ShapeOfBase>({lstm_sequence_forward_first_input_label});
    auto gather_forward_label = pattern::wrap_type<op::util::GatherBase>(
        {shapeof_forward_label, pattern::wrap_type<op::v0::Constant>(), pattern::wrap_type<op::v0::Constant>()});
    auto max_sequence_len_forward_label = pattern::wrap_type<op::v0::Constant>();
    auto broadcast_forward_label =
        pattern::wrap_type<op::v3::Broadcast>({max_sequence_len_forward_label, gather_forward_label});
    auto const_sequence_lengths_forward_label = pattern::wrap_type<op::v0::Constant>();
    auto sequence_lengths_forward_label =
        std::make_shared<pattern::op::Or>(OutputVector{broadcast_forward_label, const_sequence_lengths_forward_label});
    auto lstm_sequence_forward_label =
        pattern::wrap_type<op::v5::LSTMSequence>({lstm_sequence_forward_first_input_label,
                                                  pattern::any_input(),
                                                  pattern::any_input(),
                                                  sequence_lengths_forward_label,
                                                  pattern::any_input(),
                                                  pattern::any_input(),
                                                  pattern::any_input()});
    auto squeeze_forward_label =
        pattern::wrap_type<op::v0::Squeeze>({lstm_sequence_forward_label, pattern::wrap_type<op::v0::Constant>()},
                                            pattern::rank_equals(3));

    // backward pattern
    auto transpose_reverse_label =
        pattern::wrap_type<op::v1::Transpose, op::v1::Reshape>({data_label, pattern::wrap_type<op::v0::Constant>()});
    auto lstm_sequence_reverse_first_input_label =
        std::make_shared<pattern::op::Or>(OutputVector{transpose_reverse_label, data_label});
    auto shapeof_reverse_label = pattern::wrap_type<op::util::ShapeOfBase>({lstm_sequence_reverse_first_input_label});
    auto gather_reverse_label = pattern::wrap_type<op::util::GatherBase>(
        {shapeof_reverse_label, pattern::wrap_type<op::v0::Constant>(), pattern::wrap_type<op::v0::Constant>()});
    auto max_sequence_len_reverse_label = pattern::wrap_type<op::v0::Constant>();
    auto broadcast_reverse_label =
        pattern::wrap_type<op::v3::Broadcast>({max_sequence_len_reverse_label, gather_reverse_label});
    auto const_sequence_lengths_reverse_label = pattern::wrap_type<op::v0::Constant>();
    auto sequence_lengths_reverse_label =
        std::make_shared<pattern::op::Or>(OutputVector{broadcast_reverse_label, const_sequence_lengths_reverse_label});
    auto lstm_sequence_reverse_label =
        pattern::wrap_type<op::v5::LSTMSequence>({lstm_sequence_reverse_first_input_label,
                                                  pattern::any_input(),
                                                  pattern::any_input(),
                                                  sequence_lengths_reverse_label,
                                                  pattern::any_input(),
                                                  pattern::any_input(),
                                                  pattern::any_input()});
    auto squeeze_reverse_label =
        pattern::wrap_type<op::v0::Squeeze>({lstm_sequence_reverse_label, pattern::wrap_type<op::v0::Constant>()},
                                            pattern::rank_equals(3));

    auto concat_label = pattern::wrap_type<op::v0::Concat>({squeeze_forward_label, squeeze_reverse_label});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        auto lstm_forward = ov::as_type_ptr<op::v5::LSTMSequence>(pattern_map.at(lstm_sequence_forward_label));
        auto lstm_reverse = ov::as_type_ptr<op::v5::LSTMSequence>(pattern_map.at(lstm_sequence_reverse_label));

        NodeVector from{lstm_forward, lstm_reverse};

        if (lstm_forward->get_direction() != op::v5::LSTMSequence::direction::FORWARD ||
            lstm_reverse->get_direction() != op::v5::LSTMSequence::direction::REVERSE)
            return false;

        if (lstm_forward->get_hidden_size() != lstm_reverse->get_hidden_size())
            return false;
        if (lstm_forward->get_activations_alpha() != lstm_reverse->get_activations_alpha())
            return false;
        if (lstm_forward->get_activations_beta() != lstm_reverse->get_activations_beta())
            return false;
        if (lstm_forward->get_activations() != lstm_reverse->get_activations())
            return false;
        if (lstm_forward->get_clip() != lstm_reverse->get_clip())
            return false;

        auto squeeze_forward = pattern_map.at(squeeze_forward_label);
        if (squeeze_forward->input_value(0) != lstm_forward->output(0))
            return false;
        int64_t squeeze_forward_axis = -1;
        if (!get_constant_value(squeeze_forward->get_input_node_shared_ptr(1), squeeze_forward_axis) ||
            squeeze_forward_axis != 1)
            return false;

        auto squeeze_reverse = pattern_map.at(squeeze_reverse_label);
        if (squeeze_reverse->input_value(0) != lstm_reverse->output(0))
            return false;
        int64_t squeeze_reverse_axis = -1;
        if (!get_constant_value(squeeze_reverse->get_input_node_shared_ptr(1), squeeze_reverse_axis) ||
            squeeze_reverse_axis != 1)
            return false;

        auto concat = ov::as_type_ptr<op::v0::Concat>(pattern_map.at(concat_label));
        if (concat->get_axis() != 2)
            return false;

        from.push_back(squeeze_forward);
        from.push_back(squeeze_reverse);
        from.push_back(concat);

        bool has_input_transpose_forward = pattern_map.count(transpose_forward_label) > 0;
        bool has_input_transpose_reverse = pattern_map.count(transpose_reverse_label) > 0;
        if (has_input_transpose_forward ^ has_input_transpose_reverse)
            return false;

        bool is_forward_sequence_lengths_constant = pattern_map.count(const_sequence_lengths_forward_label) > 0;
        bool is_reverse_sequence_lengths_constant = pattern_map.count(const_sequence_lengths_reverse_label) > 0;
        if (is_forward_sequence_lengths_constant ^ is_reverse_sequence_lengths_constant)
            return false;

        if (is_forward_sequence_lengths_constant) {
            auto sequence_lengths_forward =
                ov::as_type_ptr<op::v0::Constant>(pattern_map.at(const_sequence_lengths_forward_label));
            auto sequence_lengths_reverse =
                ov::as_type_ptr<op::v0::Constant>(pattern_map.at(const_sequence_lengths_reverse_label));
            if (sequence_lengths_forward->get_shape() != sequence_lengths_reverse->get_shape())
                return false;
            auto sequence_lengths_forward_values = sequence_lengths_forward->cast_vector<int>();
            auto sequence_lengths_reverse_values = sequence_lengths_reverse->cast_vector<int>();
            if (sequence_lengths_forward_values != sequence_lengths_reverse_values)
                return false;
            from.push_back(sequence_lengths_forward);
            from.push_back(sequence_lengths_reverse);
        } else {
            auto max_sequence_len_forward =
                ov::as_type_ptr<op::v0::Constant>(pattern_map.at(max_sequence_len_forward_label));
            auto max_sequence_len_reverse =
                ov::as_type_ptr<op::v0::Constant>(pattern_map.at(max_sequence_len_reverse_label));
            if (max_sequence_len_forward->get_shape() != max_sequence_len_reverse->get_shape())
                return false;
            auto max_sequence_len_forward_values = max_sequence_len_forward->cast_vector<int>();
            auto max_sequence_len_reverse_values = max_sequence_len_reverse->cast_vector<int>();
            if (max_sequence_len_forward_values != max_sequence_len_reverse_values)
                return false;

            auto gather_forward = pattern_map.at(gather_forward_label);
            int64_t gather_index = -1;
            int64_t gather_axis = -1;
            if (!get_constant_value(gather_forward->get_input_node_shared_ptr(1), gather_index) || gather_index != 0)
                return false;
            if (!get_constant_value(gather_forward->get_input_node_shared_ptr(2), gather_axis) || gather_axis != 0)
                return false;

            auto gather_reverse = pattern_map.at(gather_reverse_label);
            gather_index = -1;
            gather_axis = -1;
            if (!get_constant_value(gather_reverse->get_input_node_shared_ptr(1), gather_index) || gather_index != 0)
                return false;
            if (!get_constant_value(gather_reverse->get_input_node_shared_ptr(2), gather_axis) || gather_axis != 0)
                return false;

            from.push_back(max_sequence_len_forward);
            from.push_back(max_sequence_len_reverse);
            from.push_back(gather_forward);
            from.push_back(gather_reverse);
        }

        NodeRegistry node_registry;

        auto new_H =
            node_registry.make<op::v0::Concat>(OutputVector{lstm_forward->input_value(1), lstm_reverse->input_value(1)},
                                               1);
        auto new_C =
            node_registry.make<op::v0::Concat>(OutputVector{lstm_forward->input_value(2), lstm_reverse->input_value(2)},
                                               1);
        auto new_W =
            node_registry.make<op::v0::Concat>(OutputVector{lstm_forward->input_value(4), lstm_reverse->input_value(4)},
                                               0);
        auto new_R =
            node_registry.make<op::v0::Concat>(OutputVector{lstm_forward->input_value(5), lstm_reverse->input_value(5)},
                                               0);
        auto new_B =
            node_registry.make<op::v0::Concat>(OutputVector{lstm_forward->input_value(6), lstm_reverse->input_value(6)},
                                               0);
        auto new_lstm = node_registry.make<op::v5::LSTMSequence>(lstm_forward->input_value(0),
                                                                 new_H,
                                                                 new_C,
                                                                 lstm_forward->input_value(3),
                                                                 new_W,
                                                                 new_R,
                                                                 new_B,
                                                                 lstm_forward->get_hidden_size(),
                                                                 op::v5::LSTMSequence::direction::BIDIRECTIONAL,
                                                                 lstm_forward->get_activations_alpha(),
                                                                 lstm_forward->get_activations_beta(),
                                                                 lstm_forward->get_activations(),
                                                                 lstm_forward->get_clip());
        if (transformation_callback(new_lstm))
            return false;

        auto transpose =
            node_registry.make<op::v1::Transpose>(new_lstm->output(0),
                                                  op::v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
        auto new_shape = node_registry.make<op::v0::Constant>(element::i32, Shape{3}, std::vector<int>{0, 0, -1});
        auto reshape = node_registry.make<op::v1::Reshape>(transpose, new_shape, true);
        reshape->set_friendly_name(concat->get_friendly_name());

        copy_runtime_info(from, node_registry.get());
        replace_node(concat, reshape);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_label, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertTensorIteratorToSequence::ConvertTensorIteratorToSequence() {
    add_matcher<ConvertTensorIteratorToLSTMSequence>();
    add_matcher<ConvertTensorIteratorToRNNSequence>();
    add_matcher<ConvertTensorIteratorToGRUSequence>();
    add_matcher<ConvertLoopToLSTMSequence>();
    add_matcher<EliminateGatherWithRange>();
    add_matcher<FuseReverseLSTMSequence>();
    add_matcher<FuseLSTMSequencesToBidirectionalLSTMSequence>();
}
