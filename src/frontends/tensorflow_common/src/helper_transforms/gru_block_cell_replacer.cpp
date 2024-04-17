// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/gru_block_cell_replacer.hpp"

#include <memory>
#include <vector>

#include "helper_ops/gru_block_cell.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::pass;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

pass::GRUBlockCellReplacer::GRUBlockCellReplacer() {
    auto gru_block_cell = pattern::wrap_type<GRUBlockCell>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;

        auto gru_block_cell_node = std::dynamic_pointer_cast<GRUBlockCell>(m.get_match_root());
        if (!gru_block_cell_node) {
            return false;
        }

        // this transformation expects only one output (the forth output) - hidden state
        // that is only output supported by OpenVINO GRUCell
        std::vector<int> restricted_output_indices = {0, 1, 2};
        for (size_t output_ind : restricted_output_indices) {
            if (gru_block_cell_node->output(output_ind).get_target_inputs().size() > 0) {
                return false;
            }
        }

        // currently, OpenVINO support only static hidden_size
        auto m_hidden_size = gru_block_cell_node->get_hidden_size();
        if (m_hidden_size.is_dynamic()) {
            return false;
        }

        auto x = gru_block_cell_node->input_value(0);
        auto h_prev = gru_block_cell_node->input_value(1);
        auto w_ru = gru_block_cell_node->input_value(2);
        auto w_c = gru_block_cell_node->input_value(3);
        auto b_ru = gru_block_cell_node->input_value(4);
        auto b_c = gru_block_cell_node->input_value(5);

        // retrive input_size and hidden_size
        auto x_shape = rg.make<v3::ShapeOf>(x, element::i64);
        auto ss_start = rg.make<v0::Constant>(element::i64, Shape{1}, 1);
        auto ss_end = rg.make<v0::Constant>(element::i64, Shape{1}, 2);
        auto ss_step = rg.make<v0::Constant>(element::i64, Shape{1}, 1);
        auto input_size = rg.make<v8::Slice>(x_shape, ss_start, ss_end, ss_step);
        auto h_prev_shape = rg.make<v3::ShapeOf>(h_prev, element::i64);
        auto hidden_size = rg.make<v8::Slice>(h_prev_shape, ss_start, ss_end, ss_step);

        // prepare weights input
        // TensorFlow provides weights in a format w_ru and w_c, where
        // z or u - update, r - reset, c or h - hidden (connection)
        // OpenVINO GRUCell accepts weights in a format w_zrh (or w_urс)
        // 1. split w_ru into w_r and w_u
        auto split_w_ru = rg.make<v1::Split>(w_ru, rg.make<v0::Constant>(element::i64, Shape{}, 1), 2);
        // 2. concatenate different parts of weights into w_zrh (or w_urс)
        auto w_urc = rg.make<v0::Concat>(OutputVector{split_w_ru->output(1), split_w_ru->output(0), w_c}, 1);

        // prepare bias in the same way
        auto split_b_ru = rg.make<v1::Split>(b_ru, rg.make<v0::Constant>(element::i64, Shape{}, 0), 2);
        auto b_urc = rg.make<v0::Concat>(OutputVector{split_b_ru->output(1), split_b_ru->output(0), b_c}, 0);

        // transpose weights
        // the current shape - [input_size + hidden_size, 3 * hidden_size]
        // we need the shape [3 * hidden_size, input_size + hidden_size]
        // in order to split WR into W and R
        auto transpose_order = rg.make<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 0});
        auto w_urc_transpose = rg.make<v1::Transpose>(w_urc, transpose_order);

        // split combined weights WR into W and R
        auto split_axis = rg.make<v0::Constant>(element::i64, Shape{}, 1);
        auto split_nums = rg.make<v0::Concat>(OutputVector{input_size, hidden_size}, 0);
        auto split_WR = rg.make<v1::VariadicSplit>(w_urc_transpose, split_axis, split_nums);

        auto gru_cell = rg.make<v3::GRUCell>(x,
                                             h_prev,
                                             split_WR->output(0),
                                             split_WR->output(1),
                                             b_urc,
                                             m_hidden_size.get_length());

        // preserve names of the node and the output tensor
        gru_cell->set_friendly_name(m.get_match_root()->get_friendly_name() + ":3");
        copy_runtime_info(gru_block_cell_node, rg.get());

        // replace GRUBlockCell with GRUCell manually instead of calling
        // ov::replace_node(m.get_match_root(), gru_cell);
        // because GRUBlockCell has 4 outputs and GRUCell has just one
        m.get_match_root()->output(3).replace(gru_cell->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(gru_block_cell, "ov::frontend::tensorflow::pass::GRUBlockCellReplacer");
    register_matcher(m, callback);
}
