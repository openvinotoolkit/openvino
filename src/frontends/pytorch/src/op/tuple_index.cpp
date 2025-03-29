// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_tuple_index(const NodeContext& context) {
    // prim::TupleIndex(Any tup, int i) -> Any
    num_inputs_check(context, 2, 2);
    auto tuple = context.get_input(0).get_node_shared_ptr();
    if (cast_fw_node(tuple, "prim::TupleConstruct")) {
        // this case require index to be constant
        auto index = context.const_input<int64_t>(1);
        PYTORCH_OP_CONVERSION_CHECK(static_cast<size_t>(index) < tuple->get_input_size(),
                                    "Index of TupleIndex operation is higher then number of tuple elements.");
        return {tuple->get_input_source_output(index)};
    } else {
        // Assume this case is when tuple is represented as tensor
        auto index = context.get_input(1);
        auto zero = v0::Constant::create(element::i32, Shape{}, {0});
        return {std::make_shared<v8::Gather>(context.get_input(0), index, zero)};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov