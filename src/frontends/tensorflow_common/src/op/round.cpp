// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/round.hpp"
#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/concat.hpp"
#include "utils.hpp"
#include "helper_ops/complex_type_mark.hpp"


using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {


OutputVector round;
OutputVector translate_round_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Round", "ROUND"});
    auto input = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    if (complex_type_mark) {
         element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_part_type->input_value(0);
        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{}, 1);
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto real = make_shared<v8::Gather>(input, gather_index_real, minus_one)->output(0);
        auto imag = make_shared<v8::Gather>(input, gather_index_imag, minus_one)->output(0);
        auto round_mode = v5::Round::RoundMode::HALF_TO_EVEN;
        auto real_part = make_shared<v5::Round>(real, round_mode);
        set_node_name(node.get_name(), real_part);
        auto imag_part = make_shared<v5::Round>(imag, round_mode);
        set_node_name(node.get_name(), imag_part);
        OutputVector concat_inputs;
        concat_inputs.push_back(real_part);
        concat_inputs.push_back(imag_part);
        auto concat = make_shared<v0::Concat>(concat_inputs, 0);

        auto complex_round = make_shared<ComplexTypeMark>(concat, complex_part_type);
        round = {complex_round->output(0)};
    } else {
        // using default round mode "half_to_even" in openvino,
        // as TF has only that mode
        auto round_mode = v5::Round::RoundMode::HALF_TO_EVEN;
        auto res = make_shared<v5::Round>(input, round_mode);
        set_node_name(node.get_name(), res);
        round = {res->output(0)};
    }
    return round;
} // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov