// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/sparse_segment_ops.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
NamedOutputVector translate_sparse_reshape_op(const ov::frontend::tensorflow::NodeContext& node) {
    default_op_checks(node, 3, {"SparseReshape"});
    auto input_indices = node.get_input(0);
    auto input_shape = node.get_input(1);
    auto new_shape = node.get_input(2);

    auto const_zero = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto const_neg_one = make_shared<v0::Constant>(element::i64, Shape{}, -1);
    auto const_axis_zero = make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto reduce_axis_one = make_shared<v0::Constant>(element::i64, Shape{1}, 1);

    auto total_elements = make_shared<v1::ReduceProd>(input_shape, const_axis_zero, true);
    auto total_elements_scalar = make_shared<v0::Squeeze>(total_elements, const_axis_zero);

    auto is_neg_one = make_shared<v1::Equal>(new_shape, const_neg_one);
    auto ones_like = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto new_shape_no_neg = make_shared<v1::Select>(is_neg_one, ones_like, new_shape);
    auto known_product = make_shared<v1::ReduceProd>(new_shape_no_neg, const_axis_zero, true);
    auto known_product_scalar = make_shared<v0::Squeeze>(known_product, const_axis_zero);
    auto inferred_dim = make_shared<v1::Divide>(total_elements_scalar, known_product_scalar);
    auto output_shape = make_shared<v1::Select>(is_neg_one, inferred_dim, new_shape);

    auto input_shape_f64 = make_shared<v0::Convert>(input_shape, element::f64);
    auto log_input_shape = make_shared<v0::Log>(input_shape_f64);
    auto cum_sum_log = make_shared<v0::CumSum>(log_input_shape, const_zero, false, false);
    auto cum_prod_f64 = make_shared<v0::Exp>(cum_sum_log);
    auto cum_prod = make_shared<v0::Convert>(
        make_shared<v5::Round>(cum_prod_f64, v5::Round::RoundMode::HALF_TO_EVEN), element::i64);
    auto input_strides = make_shared<v1::Divide>(total_elements, cum_prod);

    auto strides_unsqueezed = make_shared<v0::Unsqueeze>(input_strides, const_zero);
    auto scaled_indices = make_shared<v1::Multiply>(input_indices, strides_unsqueezed);
    auto flat_indices = make_shared<v1::ReduceSum>(scaled_indices, reduce_axis_one, true);

    auto output_shape_f64 = make_shared<v0::Convert>(output_shape, element::f64);
    auto log_output_shape = make_shared<v0::Log>(output_shape_f64);
    auto cum_sum_log_out = make_shared<v0::CumSum>(log_output_shape, const_zero, false, false);
    auto cum_prod_out_f64 = make_shared<v0::Exp>(cum_sum_log_out);
    auto cum_prod_out = make_shared<v0::Convert>(
        make_shared<v5::Round>(cum_prod_out_f64, v5::Round::RoundMode::HALF_TO_EVEN), element::i64);
    auto total_elements_out = make_shared<v1::ReduceProd>(output_shape, const_axis_zero, true);
    auto output_strides = make_shared<v1::Divide>(total_elements_out, cum_prod_out);

    auto out_strides_unsqueezed = make_shared<v0::Unsqueeze>(output_strides, const_zero);
    auto out_shape_unsqueezed = make_shared<v0::Unsqueeze>(output_shape, const_zero);
    auto divided = make_shared<v1::Divide>(flat_indices, out_strides_unsqueezed);
    auto output_indices = make_shared<v1::FloorMod>(divided, out_shape_unsqueezed);

    set_out_name(node.get_name() + ":0", output_indices);
    set_out_name(node.get_name() + ":1", output_shape);

    return {{"output_indices", output_indices}, {"output_shape", output_shape}};
}

OutputVector translate_sparse_segment_sum_op(const ov::frontend::tensorflow::NodeContext& node) {
    auto input_size = node.get_input_size();
    TENSORFLOW_OP_VALIDATION(node,
                             input_size == 3 || input_size == 4,
                             "SparseSegmentSum must have either 3 or 4 inputs.");
    auto data = node.get_input(0);
    auto indices = node.get_input(1);
    auto segment_ids = node.get_input(2);

    std::shared_ptr<ov::frontend::tensorflow::SparseSegmentSum> sparse_segment_sum = nullptr;
    if (input_size == 3) {
        sparse_segment_sum =
            make_shared<ov::frontend::tensorflow::SparseSegmentSum>(data, indices, segment_ids, node.get_decoder());

    } else {
        auto num_segments = node.get_input(3);
        sparse_segment_sum = make_shared<ov::frontend::tensorflow::SparseSegmentSum>(data,
                                                                                     indices,
                                                                                     segment_ids,
                                                                                     num_segments,
                                                                                     node.get_decoder());
    }

    set_node_name(node.get_name(), sparse_segment_sum);
    return sparse_segment_sum->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
