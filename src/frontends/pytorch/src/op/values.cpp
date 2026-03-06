// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/frontend/exception.hpp"
#include "utils.hpp"    

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_values(const NodeContext& context) {
    // aten::values(Tensor(a) self) -> Tensor(a)
    // aten::values(Dict(a) self) -> Tensor[] (list of dict values)
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto producer = input.get_node_shared_ptr();

    if (auto dict_construct = cast_fw_node(producer, "prim::DictConstruct")) {
        // Check if the input is produced by prim::DictConstruct (dict input)
        // inputs are key-value pairs, we need to return the values
        // input are stored as [key1, value1, key2, value2, ...]
        // thus we start from the second element and step by 2
        // we need to check if the inputs are divisible by 2
        // if not, we throw an error
        const auto inputs = dict_construct->input_values();
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() % 2 == 0,
                                     "aten::values: prim::DictConstruct inputs number is not divisible by 2.");
        OutputVector value_outputs;
        for (size_t i = 1; i < inputs.size(); i += 2) {
            value_outputs.push_back(inputs.at(i));
        }
        return {context.mark_node(make_list_construct(value_outputs))};
    }
    // Tensor input: aten::values(Tensor) returns the tensor itself (single value)
    // Only sparse tensor have values() method
    // dense tensor does not have values() method
    if (auto tensor = cast_fw_node(producer, {"aten::sparse_coo_tensor", "aten::sparse_csr_tensor", "aten::sparse_csc_tensor", "aten::sparse_bsr_tensor", "aten::sparse_bsc_tensor"})) {
        FRONT_END_OP_CONVERSION_CHECK(false, "aten::values: Sparse tensor input is not supported.");
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "aten::values: input is not a sparse tensor or dict.");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov