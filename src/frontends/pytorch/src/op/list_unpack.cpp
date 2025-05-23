// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_list_unpack(const NodeContext& context) {
    auto input = context.get_input(0);

    auto complex = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    bool is_complex = complex != nullptr;
    if (is_complex) {
        input = complex->get_input_source_output(0);
    }

    if (const auto& list = cast_fw_node(input.get_node_shared_ptr(), "prim::ListConstruct")) {
        // ListConstruct -> ListUnpack can be annihilated
        auto res = list->input_values();
        if (is_complex) {
            for (auto& output : res) {
                output = context.mark_node(std::make_shared<ComplexTypeMark>(output));
            }
        }
        return res;
    } else {
        const auto& outputs =
            make_framework_node(context, "Lists are not supported yet and can be resolved only in specific cases.");
        OutputVector res;
        const auto& input_node = input.get_node_shared_ptr();
        const auto& quantized_node = input_node->input_value(0);
        if (const auto& quantized_pt_node = cast_quantized_fw_node(quantized_node.get_node_shared_ptr())) {
            if (const auto& chunk_node = cast_fw_node(input_node, "aten::chunk")) {
                for (const auto& output : outputs) {
                    res.push_back(
                        context.mark_node(std::make_shared<QuantizedPtNode>(quantized_pt_node->get_type(),
                                                                            output,
                                                                            quantized_pt_node->get_scale(),
                                                                            quantized_pt_node->get_zero_point(),
                                                                            quantized_pt_node->get_dtype())));
                }
                return res;
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported operation type.");
            }
        } else {
            return outputs;
        }
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov