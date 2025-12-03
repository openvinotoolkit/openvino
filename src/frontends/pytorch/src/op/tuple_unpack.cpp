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
    auto [input, complex] = unwrap_complex(context.get_input(0));

    if (const auto& tuple = cast_fw_node(input.get_node_shared_ptr(), "prim::TupleConstruct")) {
        // TupleConstruct -> TupleUnpack can be annihilated
        auto res = tuple->input_values();
        return wrap_complex(context, res, complex);
    } else {
        // Create framework node for unresolved cases
        const auto& outputs =
            make_framework_node(context, "Tuples are not supported yet and can be resolved only in specific cases.");
        return wrap_complex(context, outputs, complex);
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
