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

OutputVector shape(const ov::frontend::tensorflow::NodeContext& node) {
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    std::map<std::string, ov::Any> attrs {
            {"out_type", get_ov_type(decoder->get_attribute(&tflite::ShapeOptions::out_type))},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_shape_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
