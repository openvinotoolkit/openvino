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
using namespace ov::opset9;
using namespace ov::frontend::tensorflow;

pass::BlockLSTMToLSTMSequenceOneOutput::BlockLSTMToLSTMSequenceOneOutput() {
    auto block_lstm = pattern::wrap_type<BlockLSTM>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto block_lstm_node = std::dynamic_pointer_cast<BlockLSTM>(m.get_match_root());
        if (!block_lstm_node) {
            return false;
        }

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

        auto block_lstm_node_name = block_lstm->get_friendly_name();
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

        // this transformation expects only one output - concatenated hidden states
        // the only output of BlockLSTM that is supported by LSTMSequence
        std::vector<int> restricted_output_indices = {0, 1, 2, 3, 4, 5};
        for (size_t output_ind : restricted_output_indices) {
            if (block_lstm_node->output(output_ind).get_target_inputs().size() > 0) {
                return false;
            }
        }

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

        // adjust output of concatenated of hidden states from LSTMSequence to have it in a format [time_len,
        // batch_size, hidden_size]
        // 1. squeeze extra dimension - num_directions
        auto squeeze_axis = rg.make<Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
        auto squeeze_output_hidden_states = rg.make<Squeeze>(lstm_sequence->output(0), squeeze_axis);
        // 2. transpose the output to rotate batch and time dimensions
        auto output_hidden_states_order = rg.make<Constant>(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto output_hidden_states = rg.make<Transpose>(squeeze_output_hidden_states, output_hidden_states_order);

        // preserve names of the node and the output tensor
        output_hidden_states->set_friendly_name(m.get_match_root()->get_friendly_name() + ":6");
        copy_runtime_info(block_lstm_node, rg.get());

        // replace BlockLSTM with LSTMSequence manually instead of calling
        // ov::replace_node(m.get_match_root(), lstm_sequence);
        // because BlockLSTM has 7 outputs and LSTMSequence has three outputs
        m.get_match_root()->output(6).replace(output_hidden_states->output(0));
        return true;
    };

    auto m =
        std::make_shared<ngraph::pattern::Matcher>(block_lstm,
                                                   "ov::frontend::tensorflow::pass::BlockLSTMToLSTMSequenceOneOutput");
    register_matcher(m, callback);
}
