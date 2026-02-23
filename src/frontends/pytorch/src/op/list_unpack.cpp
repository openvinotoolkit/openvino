// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_list_unpack(const NodeContext& context) {
    // ComplexTypeMark handling: unwrap input, create FrameworkNode with unwrapped data,
    // wrap outputs. This ensures transformations see clean graph without ComplexTypeMark.
    auto [input, complex] = unwrap_complex(context.get_input(0));

    // Check for SequenceMark first (new list/tuple construct mechanism)
    if (const auto& seq_mark = ov::as_type_ptr<SequenceMark>(input.get_node_shared_ptr())) {
        // SequenceMark -> ListUnpack can be annihilated
        auto res = seq_mark->get_sequence();
        return wrap_complex(context, res, complex);
    } else {
        // Create FrameworkNode with UNWRAPPED input (transformations will see clean graph)
        auto list_unpack_fw = std::make_shared<PtFrameworkNode>(context.get_decoder(),
                                                                OutputVector{input},  // unwrapped!
                                                                context.get_output_size());
        context.mark_node(list_unpack_fw);
        add_exception_to_fw_node(list_unpack_fw,
                                 "Lists are not supported yet and can be resolved only in specific cases.");

        auto outputs = list_unpack_fw->outputs();

        // Handle quantized nodes
        const auto& input_node = input.get_node_shared_ptr();
        if (input_node->get_input_size() > 0) {
            const auto& quantized_node = input_node->input_value(0);
            if (const auto& quantized_pt_node = cast_quantized_fw_node(quantized_node.get_node_shared_ptr())) {
                if (const auto& chunk_node = cast_fw_node(input_node, "aten::chunk")) {
                    OutputVector res;
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
            }
        }

        // Wrap outputs in ComplexTypeMark if input was complex
        return wrap_complex(context, outputs, complex);
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov