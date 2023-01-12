// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector squeeze(const ov::frontend::tensorflow::NodeContext& node) {
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    auto data = node.get_input(0);
    auto squeeze_dims = decoder->get_attribute(&tflite::SqueezeOptions::squeeze_dims);
    std::vector<int64_t> axes {squeeze_dims->begin(), squeeze_dims->end()};
    return attribute_helper(node, {{"axis", axes}}, ov::frontend::tensorflow::op::translate_squeeze_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
