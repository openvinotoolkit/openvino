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

OutputVector batch_matmul(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    std::map<std::string, ov::Any> attrs{
        {"adj_x", decoder->get_attribute(&tflite::BatchMatMulOptions::adj_x)},
        {"adj_y", decoder->get_attribute(&tflite::BatchMatMulOptions::adj_y)},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_batch_mat_mul_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
