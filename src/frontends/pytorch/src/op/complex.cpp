// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_view_as_real(const NodeContext& context) {
    num_inputs_check(context, 1, 1, true);
    auto complex = context.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());
    PYTORCH_OP_CONVERSION_CHECK(complex_type_mark, "aten::view_as_real is only supported for complex tensors");

    return {complex_type_mark->get_data()};
};

OutputVector translate_view_as_complex(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto complex = context.get_input(0);

    return {context.mark_node(std::make_shared<ComplexTypeMark>(complex))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
