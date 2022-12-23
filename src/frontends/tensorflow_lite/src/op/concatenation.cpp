// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "utils.hpp"
#include "op_translation_utils.hpp"


using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector concatenation(const ov::frontend::tensorflow::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    const auto& decoder = node.get_decoder();
    const auto* concat_opts = decoder->get_attribute("ConcatenationOptions").as<const tflite::ConcatenationOptions*>();
    const std::map<std::string, ov::Any> attrs {
            {"axis", static_cast<int64_t>(concat_opts->axis())},
            {"activation", EnumNameActivationFunctionType(concat_opts->fused_activation_function())},
    };
    auto decoder_for_tf_translator = std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(decoder, attrs, "tflite::CONCATENATION", true);
    ov::OutputVector inputs(node.get_input_size());
    for (auto i = 0; i < node.get_input_size(); ++i) {
        inputs[i] = node.get_input(i);
    }
    auto context = ov::frontend::tensorflow::NodeContext(decoder_for_tf_translator, inputs);
    auto output = ov::frontend::tensorflow::op::translate_concat_op(context);
    del_output_names(output);
    get_activation(output, node, decoder_for_tf_translator);
    del_output_names(output);
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov



