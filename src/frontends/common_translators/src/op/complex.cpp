// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace common_translators {

OutputVector translate_complex(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto real = context.get_input(0);
    auto imag = context.get_input(1);

    auto complex_mark = context.mark_node(make_shared<ComplexTypeMark>(real, imag));

    return {complex_mark};
};

OutputVector translate_real(const NodeContext& context) {
    num_inputs_check(context, 1, 1, true);
    auto complex = context.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());
    if (!complex_type_mark)
        // if input is not a complex number, just return it as is. This is allowed in torch.
        return {complex};
    return {complex_type_mark->get_real()};
};

OutputVector translate_imag(const NodeContext& context) {
    num_inputs_check(context, 1, 1, true);
    auto complex = context.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());
    auto op_type = context.get_op_type();
    FRONT_END_OP_CONVERSION_CHECK(complex_type_mark, op_type + " operation expects complex type tensor on input.");

    return {complex_type_mark->get_imag()};
};

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
