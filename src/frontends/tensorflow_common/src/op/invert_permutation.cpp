// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_invert_permutation_op(const NodeContext& node) {
    default_op_checks(node, 1, {"InvertPermutation"});
    auto x = node.get_input(0);
    auto node_name = node.get_name();

    // compute a number of elements in x
    // by definition, x is 1D tensor
    auto x_shape = make_shared<v3::ShapeOf>(x, element::i64);
    auto squeeze_dim = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto n = make_shared<v0::Squeeze>(x_shape, squeeze_dim);

    // generate a range [0, n)
    auto zero = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto one = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto values = make_shared<v4::Range>(zero, n, one, element::i64)->output(0);
    values = make_shared<v1::ConvertLike>(values, x);

    // compute inverted permutation
    auto axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto invert_permutation = make_shared<v3::ScatterUpdate>(x, x, values, axis);

    set_node_name(node_name, invert_permutation);
    return {invert_permutation};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
