// Copyright (C) 2018-2022 Intel Corporation
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

OutputVector concatenation(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    int64_t axis = static_cast<int64_t>(decoder->get_attribute(&tflite::ConcatenationOptions::axis));
    auto concat = make_shared<opset10::Concat>(node.get_inputs(), axis);
    concat->set_friendly_name(decoder->get_op_name());
    return concat->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
