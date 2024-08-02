// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_matrix_band_part_op(const NodeContext& node) {
    default_op_checks(node, 3, {"MatrixBandPart"});

    // Input tensor and parameters
    auto input = node.get_input(0);
    auto num_lower = node.get_input(1);
    auto num_upper = node.get_input(2);

    // create scalar auxiliary constants
    auto const_zero = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto const_one = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto const_two = make_shared<v0::Constant>(element::i64, Shape{}, 2);

    // input has a shape [I, J, K, ..., M, N]
    // compute sizes of two last dimensions of M and N
    auto input_shape = make_shared<v3::ShapeOf>(input, element::i64);
    auto input_rank = make_shared<v3::ShapeOf>(input_shape, element::i64);
    auto input_rank_minus_one = make_shared<v1::Subtract>(input_rank, const_one);
    auto input_rank_minus_two = make_shared<v1::Subtract>(input_rank, const_two);
    auto slice_step = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto slice_axis = make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto m = make_shared<v8::Slice>(input_shape, input_rank_minus_two, input_rank_minus_one, slice_step, slice_axis)
                 ->output(0);
    auto n = make_shared<v8::Slice>(input_shape, input_rank_minus_one, input_rank, slice_step, slice_axis)->output(0);

    // generate ranges [0, M) and [0, N)
    auto scalar_shape = make_shared<v0::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    m = make_shared<v1::Reshape>(m, scalar_shape, false);
    n = make_shared<v1::Reshape>(n, scalar_shape, false);
    auto range_m = make_shared<v4::Range>(const_zero, m, const_one, element::i64)->output(0);
    auto range_n = make_shared<v4::Range>(const_zero, n, const_one, element::i64)->output(0);
    range_m = make_shared<v0::Unsqueeze>(range_m, const_one);
    range_n = make_shared<v0::Unsqueeze>(range_n, const_zero);

    // adjust num_lower and num_upper to have them of type i64
    // the same as M and N
    // it is needed for in_band computation
    num_lower = make_shared<v0::Convert>(num_lower, element::i64);
    num_upper = make_shared<v0::Convert>(num_upper, element::i64);

    // compute in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) && (num_upper < 0 || (n-m) <= num_upper)
    auto num_lower_less_zero = make_shared<v1::Less>(num_lower, const_zero);
    auto i_minus_j = make_shared<v1::Subtract>(range_m, range_n);
    auto i_minus_j_less_eq_num_lower = make_shared<v1::LessEqual>(i_minus_j, num_lower);
    auto num_upper_less_zero = make_shared<v1::Less>(num_upper, const_zero);
    auto j_minus_i = make_shared<v1::Subtract>(range_n, range_m);
    auto j_minus_i_less_eq_num_upper = make_shared<v1::LessEqual>(j_minus_i, num_upper);
    auto in_band1 = make_shared<v1::LogicalOr>(num_lower_less_zero, i_minus_j_less_eq_num_lower);
    auto in_band2 = make_shared<v1::LogicalOr>(num_upper_less_zero, j_minus_i_less_eq_num_upper);
    auto in_band = make_shared<v1::LogicalAnd>(in_band1, in_band2);

    // create zero constant of the same type as input
    auto zero = create_same_type_const_scalar<int32_t>(input, 0);

    auto result = make_shared<v1::Select>(in_band, input, zero);

    set_node_name(node.get_name(), result);
    return {result};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
