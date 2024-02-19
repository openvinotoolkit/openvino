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

OutputVector cast(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    std::map<std::string, ov::Any> attrs{
        {"DstT", decoder->get_output_tensor_type(0)},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_cast_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
