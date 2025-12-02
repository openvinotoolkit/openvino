// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_tuple_unpack(const NodeContext& context) {
    auto input = context.get_input(0);

    // Check if input is wrapped in ComplexTypeMark
    auto complex = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    bool is_complex = complex != nullptr;
    if (is_complex) {
        input = complex->get_input_source_output(0);
    }

    if (const auto& tuple = cast_fw_node(input.get_node_shared_ptr(), "prim::TupleConstruct")) {
        // TupleConstruct -> TupleUnpack can be annihilated
        auto res = tuple->input_values();
        // Preserve ComplexTypeMark for complex tensor outputs
        if (is_complex) {
            for (auto& output : res) {
                output = context.mark_node(std::make_shared<ComplexTypeMark>(output));
            }
        }
        return res;
    } else {
        // Create framework node for unresolved cases
        const auto& outputs =
            make_framework_node(context, "Tuples are not supported yet and can be resolved only in specific cases.");
        // Preserve ComplexTypeMark for complex tensor outputs
        if (is_complex) {
            OutputVector complex_outputs;
            for (const auto& output : outputs) {
                complex_outputs.push_back(context.mark_node(std::make_shared<ComplexTypeMark>(output)));
            }
            return complex_outputs;
        }
        return outputs;
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
