// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_squeeze(const NodeContext& context) {
    num_inputs_check(context, 1, 2, true);  // allow_complex = true
    auto [data, complex] = unwrap_complex(context.get_input(0));

    std::shared_ptr<Node> res;
    if (context.input_is_none(1)) {
        res = context.mark_node(std::make_shared<v0::Squeeze>(data));
    } else {
        auto dim = context.get_input(1);
        if (complex) {
            if (dim.get_element_type() != element::i32) {
                dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
            }
            auto rank = std::get<1>(get_shape_rank(context, context.get_input(0), true));
            dim = normalize_axis(context, dim, rank);
        }
        res = context.mark_node(std::make_shared<v0::Squeeze>(data, dim));
    }

    return {wrap_complex(context, res->output(0), complex)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
