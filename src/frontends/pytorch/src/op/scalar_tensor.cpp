// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_scalar_tensor_fx(const NodeContext& context) {
    // aten.scalar_tensor.default(-100.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    // TorchScript: aten::scalar_tensor(Scalar s, *, ScalarType? dtype, Layout? layout, Device? device, bool?
    // pin_memory)
    num_inputs_check(context, 1, 5);
    auto data = context.get_input(0);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        data = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), dtype));
    } else if (context.get_input_size() > 1 && !context.input_is_none(1)) {
        data = apply_dtype(context, 1, data);
    }
    // layout and device can be ignored
    return {data};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
