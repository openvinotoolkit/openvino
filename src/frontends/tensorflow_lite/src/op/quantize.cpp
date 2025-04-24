// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "tflite_ops/tflite_quantize.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector quantize(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto convert = make_shared<opset10::Convert>(node.get_input(0), element::f32);
    disable_constant_folding(convert);
    convert->set_friendly_name(node.get_name());
    return convert->outputs();
}

OutputVector dequantize(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto decoder = std::dynamic_pointer_cast<DecoderBaseOperation>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder, "Operation decoder is expected in dequantize translator");
    auto convert = make_shared<opset10::Convert>(node.get_input(0), decoder->get_output_tensor_type(0));
    disable_constant_folding(convert);
    convert->set_friendly_name(node.get_name());
    return convert->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
