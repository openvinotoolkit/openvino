// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
using namespace std;

OutputVector translate_abs(const NodeContext& context) {
    num_inputs_check(context, 1, 2, true);

    auto res = ComplexTypeMark::abs(context, context.get_input(0));
    auto out_type = context.get_output_type(0);
    if (out_type.is<element::Type>()) {
        auto dtype = out_type.as<element::Type>();
        if (dtype.is_static() && dtype != res.get_element_type()) {
            res = context.mark_node(std::make_shared<ov::op::v0::Convert>(res, dtype));
        }
    }
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
