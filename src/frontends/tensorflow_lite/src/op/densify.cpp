// Copyright (C) 2018-2023 Intel Corporation
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

OutputVector densify(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    std::map<std::string, ov::Any> attrs{
        {"output_type", get_ov_type(decoder->get_attribute(&tflite::ArgMinOptions::output_type))},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_arg_min_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
