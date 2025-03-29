// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_unravel_index_op(const NodeContext& node) {
    default_op_checks(node, 2, {"UnravelIndex"});
    auto indices = node.get_input(0);
    auto dims = node.get_input(1);
    auto node_name = node.get_name();

    // create auxiliary constant
    auto const_one_same_type = make_shared<v0::Constant>(element::i32, Shape{1}, 1)->output(0);
    const_one_same_type = make_shared<v1::ConvertLike>(const_one_same_type, dims);
    auto num_dims = make_shared<v3::ShapeOf>(dims, element::i32)->output(0);
    auto const_zero = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);

    // generate upper triangular matrice from dims
    // for example, if dims = [3, 4, 5] it generates the following matrice:
    // [3 4 5]
    // [1 4 5]
    // [1 1 5]
    // 1. unsqueeze dims to have it of a shape [1, n]
    dims = make_shared<v0::Unsqueeze>(dims, const_zero);
    // 2. create a constant of ones with a shape [n, 1]
    auto shape_n1 = make_shared<v0::Concat>(OutputVector{num_dims, const_one}, 0);
    auto const_one_n1 = make_shared<v3::Broadcast>(const_one_same_type, shape_n1);
    // 3. generate a mask to have upper triangular matric
    auto scalar_zero = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto scalar_num = make_shared<v0::Squeeze>(num_dims, const_zero);
    auto scalar_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto rng0n = make_shared<v4::Range>(scalar_zero, scalar_num, scalar_one, element::i32);
    auto rng0n_1n = make_shared<v0::Unsqueeze>(rng0n, const_zero);
    auto rng0n_n1 = make_shared<v0::Unsqueeze>(rng0n, const_one);
    auto mask = make_shared<v1::LessEqual>(rng0n_n1, rng0n_1n);
    // 4. generate the upper triangular matrice
    auto upper_trig_matrice = make_shared<v1::Select>(mask, dims, const_one_n1);

    // compute reduce prod to understand how many elements  are place in each slice
    // for example, if dims = [3, 4, 5], slice by highest dimension has 20 elements
    // lower dimension has 5 elements, etc.
    // this way it computes [60, 20, 5] where 60 is a number of all elements in example tensor
    auto num_elems_by_slice = make_shared<v1::ReduceProd>(upper_trig_matrice, const_one, false)->output(0);

    // pad the resulted product with one and exclude the first element in the product
    // the number of elements in the whole tensor is not needed
    // for example, it computes div_coeffs = [20, 5, 1] and mod_coeffs = [60, 20, 5] by shifting
    auto coeffs = make_shared<v0::Concat>(OutputVector{num_elems_by_slice, const_one_same_type}, 0);
    auto stop_slice = make_shared<v0::Constant>(ov::element::i32, Shape{1}, numeric_limits<int>::max());
    auto div_coeffs = make_shared<v8::Slice>(coeffs, const_one, stop_slice, const_one)->output(0);
    auto mod_coeffs = num_elems_by_slice;

    // using computed div_coeffs and mod_coeffs, compute indices of each element by its index in the flattened tensor
    // the resulted reduce product will be used for indices computation
    // for example, the product is a vector
    // each index will be computed by formula: (index % mod_coeff) / div_coeff
    indices = make_shared<v0::Unsqueeze>(indices, const_zero);
    div_coeffs = make_shared<v0::Unsqueeze>(div_coeffs, const_one);
    mod_coeffs = make_shared<v0::Unsqueeze>(mod_coeffs, const_one);
    auto result_indices = make_shared<v1::FloorMod>(indices, mod_coeffs)->output(0);
    result_indices = make_shared<v1::Divide>(result_indices, div_coeffs);
    set_node_name(node_name, result_indices.get_node_shared_ptr());

    return {result_indices};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
