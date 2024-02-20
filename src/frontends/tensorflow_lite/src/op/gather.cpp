// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector gather(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    auto batch_dims = static_cast<int64_t>(decoder->get_attribute(&tflite::GatherOptions::batch_dims));
    auto axis = opset10::Constant::create(element::i32, {}, {decoder->get_attribute(&tflite::GatherOptions::axis)});
    auto input = node.get_input(0);
    auto input_indices = node.get_input(1);
    auto res = make_shared<opset10::Gather>(input, input_indices, axis, batch_dims);
    res->set_friendly_name(node.get_name());
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
