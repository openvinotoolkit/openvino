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

OutputVector depth_to_space(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    std::map<std::string, ov::Any> attrs{
        {"block_size", static_cast<int64_t>(decoder->get_attribute(&tflite::DepthToSpaceOptions::block_size))},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_depth_to_space_op, "DepthToSpace");
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
