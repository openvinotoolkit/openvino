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

OutputVector squeeze(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    auto squeeze_dims = decoder->get_attribute(&tflite::SqueezeOptions::squeeze_dims);
    std::vector<int64_t> axes{squeeze_dims->begin(), squeeze_dims->end()};
    return attribute_helper(node, {{"axis", axes}}, ov::frontend::tensorflow::op::translate_squeeze_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
