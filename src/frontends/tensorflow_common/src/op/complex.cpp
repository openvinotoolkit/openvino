// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_complex_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Complex"}, true);
    auto real = node.get_input(0);
    auto imag = node.get_input(1);
    auto tout = node.get_attribute<string>("Tout", "DT_COMPLEX64");
    element::Type complex_part_type = (tout == "DT_COMPLEX64" ? element::f32 : element::f64);

    // compute target shape to which real and imag parts must be broadcasted
    // and broadcast them
    auto real_shape = make_shared<v3::ShapeOf>(real, element::i32);
    auto imag_shape = make_shared<v3::ShapeOf>(imag, element::i32);
    auto target_shape = compute_broadcast_args(real_shape, imag_shape);
    real = make_shared<v3::Broadcast>(real, target_shape);
    imag = make_shared<v3::Broadcast>(imag, target_shape);

    // expand real and imaginary parts with one dimension in the end for further concatenation
    // this way, complex tensor with real and imag of shapes [N1, N2, ..., Nk] will be represented as floating-point
    // tensor of shape [N1, N2, ..., Nk, 2]
    auto real_rank = compute_subgraph_scalar_rank(real, element::i32, false);
    real = make_shared<v0::Unsqueeze>(real, real_rank);
    imag = make_shared<v0::Unsqueeze>(imag, real_rank);

    // concatenate real and imaginary parts to have a complex tensor represented as a floating-point tensor of shape
    // [N1, N2, ..., Nk, 2]
    auto complex_tensor = make_shared<v0::Concat>(OutputVector{real, imag}, -1)->output(0);
    complex_tensor = make_shared<v0::Convert>(complex_tensor, complex_part_type);

    // set node name and tensor
    set_node_name(node.get_name(), complex_tensor.get_node_shared_ptr());

    // create complex type mark operation for upcoming operations in a graph
    auto complex_type_mark = make_shared<ComplexTypeMark>(complex_tensor, complex_part_type);
    return complex_type_mark->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
