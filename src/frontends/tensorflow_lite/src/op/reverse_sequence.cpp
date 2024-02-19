// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::frontend::tensorflow::op;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector reverse_sequence(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    std::map<std::string, ov::Any> attrs{
        {"seq_dim", static_cast<int64_t>(decoder->get_attribute(&tflite::ReverseSequenceOptions::seq_dim))},
        {"batch_dim", static_cast<int64_t>(decoder->get_attribute(&tflite::ReverseSequenceOptions::batch_dim))},
    };
    return attribute_helper(node, attrs, translate_reverse_sequence_op, "ReverseSequence");
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
