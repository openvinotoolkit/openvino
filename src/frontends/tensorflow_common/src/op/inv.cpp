// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_inv_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Inv"}, true);
    auto x = node.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    if (complex_type_mark) {
        x = complex_type_mark->input_value(0);
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();

        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{}, 1);

        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);

        auto x_real = make_shared<v8::Gather>(x, gather_index_real, minus_one)->output(0);
        auto x_imag = make_shared<v8::Gather>(x, gather_index_imag, minus_one)->output(0);

        auto scale =
            make_shared<v1::Add>(make_shared<v1::Multiply>(x_real, x_real), make_shared<v1::Multiply>(x_imag, x_imag));

        auto y_real = make_shared<v1::Divide>(x_real, scale);
        auto y_imag = make_shared<v1::Divide>(make_shared<v0::Negative>(x_imag), scale);

        auto real_unsqueeze = make_shared<v0::Unsqueeze>(y_real, minus_one);
        auto imag_unsqueeze = make_shared<v0::Unsqueeze>(y_imag, minus_one);

        auto concat_result = make_shared<v0::Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);
        set_node_name(node.get_name(), concat_result);

        auto complex_result = make_shared<ComplexTypeMark>(concat_result->output(0), complex_part_type);
        return {complex_result};
    }

    // prepare auxiliary one constants of the same type as the inputs
    auto one = create_same_type_const_scalar<int32_t>(x, 1);

    auto inv = make_shared<v1::Divide>(one, x);
    set_node_name(node.get_name(), inv);
    return inv->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov