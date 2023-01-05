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

OutputVector pad(const ov::frontend::tensorflow::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    string new_pad_type = "Pad";
    auto decoder = std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(node.get_decoder(),
                                                                               std::map<std::string, ov::Any>{},
                                                                               new_pad_type);
    auto inputs = OutputVector{node.get_input(0), node.get_input(1)};
    auto context = ov::frontend::tensorflow::NodeContext(decoder, inputs);
    auto output = ov::frontend::tensorflow::op::translate_pad_op(context);
    del_output_names(output);
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
