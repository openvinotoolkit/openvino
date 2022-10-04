// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/block_lstm_replacer.hpp"

#include <memory>
#include <vector>

#include "helper_ops/block_lstm.hpp"
#include "ngraph/rt_info.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::pass;
using namespace ov::pass::pattern;
using namespace ov::opset9;
using namespace ov::frontend::tensorflow;

namespace {
std::function<bool(ov::Output<ov::Node>)> can_have_outputs(const std::vector<size_t>& allowed_output_indices) {
    return [=](ov::Output<ov::Node> output) -> bool {
        auto block_lstm_node = output.get_node_shared_ptr();
        auto output_size = block_lstm_node->get_output_size();
        for (size_t output_ind = 0; output_ind < output_size; ++output_ind) {
            if (std::find(allowed_output_indices.begin(), allowed_output_indices.end(), output_ind) !=
                allowed_output_indices.end()) {
                continue;
            }
            if (block_lstm_node->output(output_ind).get_target_inputs().size() > 0) {
                return false;
            }
        }
        return true;
    };
}
}  // namespace

pass::BlockLSTMReplacer::BlockLSTMReplacer() {
    // Pattern 1: BlockLSTM with last state cell output (BlockLSTM -> Concat -> GatherND)
    // used in DeepSpeech model
    auto block_lstm_1 = pattern::wrap_type<BlockLSTM>(can_have_outputs({1, 6}));
    auto states_cell_1 = pattern::wrap_type<Concat>({pattern::any_input(), block_lstm_1});
    auto pattern1 = pattern::wrap_type<GatherND>({states_cell_1, pattern::any_input()});

    // Pattern 2: BlockLSTM with just one output, concatenated hidden states (BlockLSTM)
    auto pattern2 = pattern::wrap_type<BlockLSTM>(can_have_outputs({6}));

    auto root = std::make_shared<pattern::op::Or>(OutputVector{pattern1, pattern2});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto is_pattern1 = (pattern_map.find(pattern1) != std::end(pattern_map));
        auto is_pattern2 = (pattern_map.find(pattern2) != std::end(pattern_map));

        // find for each pattern BlockLSTM node for which we adjust inputs
        // and check its attributes before the transformation
        std::shared_ptr<BlockLSTM> block_lstm_node;
        std::shared_ptr<Node> last_state_c_node;
        ov::NodeVector rt_info_from;
        if (is_pattern1) {
            block_lstm_node = std::dynamic_pointer_cast<BlockLSTM>(pattern_map.at(block_lstm_1));
            auto concat_node = std::dynamic_pointer_cast<Concat>(pattern_map.at(states_cell_1));
            if (!concat_node || concat_node->get_axis() != 0) {
                // timestep is the first dimension
                return false;
            }
            last_state_c_node = pattern_map.at(pattern1);
            rt_info_from = {block_lstm_node, concat_node, last_state_c_node};
        } else if (is_pattern2) {
            block_lstm_node = std::dynamic_pointer_cast<BlockLSTM>(pattern_map.at(pattern2));
            rt_info_from = {block_lstm_node};
        }
        if (!block_lstm_node) {
            return false;
        }

        NodeRegistry rg;
        // currently, LSTMSequence does not support peephole and cell clip
        if (block_lstm_node->get_use_peephole()) {
            return false;
        }
        if (block_lstm_node->get_cell_clip() != -1.0f) {
            return false;
        }

        // currently, OpenVINO support only static hidden_size
        // since this is an attribute of LSTMSequence operation
        auto hidden_size = block_lstm_node->get_hidden_size();
        if (hidden_size.is_dynamic()) {
            return false;
        }

        auto block_lstm_node_name = block_lstm_node->get_friendly_name();
        auto seq_len_max = block_lstm_node->input_value(0);
        auto x = block_lstm_node->input_value(1);
        auto cs_prev = block_lstm_node->input_value(2);
        auto h_prev = block_lstm_node->input_value(3);
        auto weights = block_lstm_node->input_value(4);
        auto wci = block_lstm_node->input_value(5);
        auto wcf = block_lstm_node->input_value(6);
        auto wco = block_lstm_node->input_value(7);
        auto bias = block_lstm_node->input_value(8);

        // retrieve input_size
        auto x_shape = rg.make<ShapeOf>(x, element::i64);
        auto ss_start = rg.make<Constant>(element::i64, Shape{1}, 2);
        auto ss_stop = rg.make<Constant>(element::i64, Shape{1}, 3);
        auto ss_step = rg.make<Constant>(element::i64, Shape{1}, 1);
        auto input_size = rg.make<StridedSlice>(x_shape,
                                                ss_start,
                                                ss_stop,
                                                ss_step,
                                                std::vector<int64_t>{0},
                                                std::vector<int64_t>{0});

        // retrieve the batch size
        // now x is in a format [time_len, batch_size, input_size]
        auto ss_start2 = rg.make<Constant>(element::i64, Shape{1}, 1);
        auto ss_stop2 = rg.make<Constant>(element::i64, Shape{1}, 2);
        auto batch_size = rg.make<StridedSlice>(x_shape,
                                                ss_start2,
                                                ss_stop2,
                                                ss_step,
                                                std::vector<int64_t>{0},
                                                std::vector<int64_t>{0});

        auto hidden_size_const =
            rg.make<Constant>(element::i64, Shape{1}, std::vector<int64_t>{hidden_size.get_length()});

        // adjust weights and bias
        // 1. reshape weights and bias to highlight channel dimension
        auto new_weight_shape = rg.make<Constant>(element::i64, Shape{3}, std::vector<int64_t>{0, 4, -1});
        auto weight_reshape = rg.make<Reshape>(weights, new_weight_shape, true);
        auto new_bias_shape = rg.make<Constant>(element::i64, Shape{2}, std::vector<int64_t>{4, -1});
        auto bias_reshape = rg.make<Reshape>(bias, new_bias_shape, true);
        // 2. reorder gates icfo --> fico for both weights and biases
        auto reorder_const = rg.make<Constant>(element::i64, Shape{4}, std::vector<int64_t>{2, 0, 1, 3});
        auto weights_axis = rg.make<Constant>(element::i64, Shape{}, 1);
        auto weights_reorder = rg.make<Gather>(weight_reshape, reorder_const, weights_axis);
        auto bias_axis = rg.make<Constant>(element::i64, Shape{}, 0);
        auto bias_reorder = rg.make<Gather>(bias_reshape, reorder_const, bias_axis);
        // 3. shift_const.value should be added to the first 1 / 4th part of the biases(f - gate : 0)
        auto shift_const = rg.make<Constant>(element::f32, Shape{}, block_lstm_node->get_forget_bias());
        auto bias_split_lens = rg.make<Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 3});
        auto bias_split = rg.make<VariadicSplit>(bias_reorder, bias_axis, bias_split_lens);
        auto bias_first_shift = rg.make<Add>(bias_split->output(0), shift_const);
        auto bias_shift = rg.make<Concat>(OutputVector{bias_first_shift, bias_split->output(1)}, 0);
        // 4. return to the original shapes
        auto new_weight_shape2 = rg.make<Constant>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
        auto weight_reshape2 = rg.make<Reshape>(weights_reorder, new_weight_shape2, true);
        // 5. normalize weights and bias
        auto transpose_order = rg.make<Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 0});
        auto new_bias_shape2 = rg.make<Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
        auto weights_normalize = rg.make<Transpose>(weight_reshape2, transpose_order);
        auto bias_normalized = rg.make<Reshape>(bias_shift, new_bias_shape2, true);
        // 6. split weights into W and R inputs
        auto WR_split_axis = rg.make<Constant>(element::i64, Shape{}, 1);
        auto WR_split_lens = rg.make<Concat>(OutputVector{input_size, hidden_size_const}, 0);
        auto WR_split = rg.make<VariadicSplit>(weights_normalize, WR_split_axis, WR_split_lens);
        // 7. unsqueeze weights and bias to have a dimension for a number of directions
        auto num_direct_axis = rg.make<Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
        auto W = rg.make<Unsqueeze>(WR_split->output(0), num_direct_axis);
        auto R = rg.make<Unsqueeze>(WR_split->output(1), num_direct_axis);
        auto B = rg.make<Unsqueeze>(bias_normalized, num_direct_axis);

        // normalize initial hidden and cell states
        auto unsqueeze_axis = rg.make<Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
        auto init_hidden_state = rg.make<Unsqueeze>(h_prev, unsqueeze_axis);
        auto init_cell_state = rg.make<Unsqueeze>(cs_prev, unsqueeze_axis);

        // prepare sequence length input for LSTMSequence
        auto seq_len_max_adjusted = rg.make<Broadcast>(seq_len_max, batch_size);

        // prepare input data since LSTMSequence accept it in a format [batch_size, time_len, input_size]
        auto x_order = rg.make<Constant>(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto x_adjusted = rg.make<Transpose>(x, x_order);

        // create LSTMSequence node and reconnect inputs and normalized weights and bias
        auto lstm_sequence = rg.make<LSTMSequence>(x_adjusted,
                                                   init_hidden_state,
                                                   init_cell_state,
                                                   seq_len_max_adjusted,
                                                   W,
                                                   R,
                                                   B,
                                                   hidden_size.get_length(),
                                                   LSTMSequence::direction::FORWARD);

        if (block_lstm_node->output(1).get_target_inputs().size() > 0) {
            // adjust output with the last state cell and connect to the main graph
            // squeeze extra dimension - num_directions
            auto squeeze_axis = rg.make<Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
            auto squeeze_last_state_cell = rg.make<Squeeze>(lstm_sequence->output(2), squeeze_axis);

            // preserve names of the node and the output tensor
            squeeze_last_state_cell->set_friendly_name(last_state_c_node->get_friendly_name());

            ov::replace_node(last_state_c_node, squeeze_last_state_cell);
        }

        if (block_lstm_node->output(6).get_target_inputs().size() > 0) {
            // adjust output of concatenated of hidden states from LSTMSequence
            // to have it in a format [time_len, batch_size, hidden_size]
            // 1. squeeze extra dimension - num_directions
            auto squeeze_axis = rg.make<Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
            auto squeeze_output_hidden_states = rg.make<Squeeze>(lstm_sequence->output(0), squeeze_axis);
            // 2. transpose the output to rotate batch and time dimensions
            auto output_hidden_states_order = rg.make<Constant>(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2});
            auto output_hidden_states = rg.make<Transpose>(squeeze_output_hidden_states, output_hidden_states_order);

            // preserve names of the node and the output tensor
            output_hidden_states->set_friendly_name(block_lstm_node->get_friendly_name() + ":6");

            // replace BlockLSTM with LSTMSequence manually instead of calling
            // ov::replace_node(m.get_match_root(), lstm_sequence);
            // because BlockLSTM has 7 outputs and LSTMSequence has three outputs
            block_lstm_node->output(6).replace(output_hidden_states->output(0));
        }

        copy_runtime_info(rt_info_from, rg.get());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, "ov::frontend::tensorflow::pass::BlockLSTMReplacer");
    register_matcher(m, callback);
}
