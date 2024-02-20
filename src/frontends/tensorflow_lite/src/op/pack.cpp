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

OutputVector pack(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    const std::map<std::string, ov::Any> attrs{
        {"axis", static_cast<int64_t>(decoder->get_attribute(&tflite::PackOptions::axis))},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_pack_op, "Pack");
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
