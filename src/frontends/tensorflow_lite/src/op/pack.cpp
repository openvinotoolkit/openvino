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

OutputVector pack(const ov::frontend::tensorflow_lite::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    const std::map<std::string, ov::Any> attrs{
        {"axis", static_cast<int64_t>(decoder->get_attribute(&tflite::PackOptions::axis))},
    };
    auto decoder_for_tf_translator =
        std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(decoder, attrs, "Pack", false);
    ov::OutputVector inputs(node.get_input_size());
    for (auto i = 0; i < node.get_input_size(); ++i) {
        inputs[i] = node.get_input(i);
    }
    auto context = ov::frontend::tensorflow_lite::NodeContext(decoder_for_tf_translator, inputs);
    auto output = ov::frontend::tensorflow::op::translate_pack_op(context);
    del_output_names(output);
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
