// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/augru_cell_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/augru_cell.hpp"

using namespace std;
using namespace ov::element;
using namespace ov::pass::pattern;

// The 1st input to the Add op is automatically broadcasted
// from 1d to 2d tensor, but to be compatible with what
// the transformation code expectes we have to broadcast the
// input manually from 1d to 2d using an Unsqueeze operation.
static std::shared_ptr<ov::Node> get_bias_add(const std::shared_ptr<ov::Node>& bias_add, ov::pass::NodeRegistry& rg) {
    auto input_source_1_ps = bias_add->input_value(1).get_partial_shape();
    if (input_source_1_ps.is_static() && input_source_1_ps.rank().get_length() == 1) {
        auto unsqueeze =
            rg.make<ov::op::v0::Unsqueeze>(bias_add->input_value(1),
                                           ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
        bias_add->input(1).replace_source_output(unsqueeze);
    }

    return bias_add;
}

// Originally, the transformation code expects
// the 1st input to be transposed via the attribute
// of Matmul 'transpose_b' (hence the input tensor of the 1st
// input is in the specific shape). However, in the
// newer version of the model it doesn't seem to be the case,
// so we need to insert a Transpose operation to make it
// compatible with the code of the transformation.
static std::shared_ptr<ov::Node> get_weights_matmul(const std::shared_ptr<ov::Node>& mat_mul,
                                                    ov::pass::NodeRegistry& rg) {
    if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(mat_mul)) {
        if (!matmul->get_transpose_b()) {
            auto transpose =
                rg.make<ov::op::v1::Transpose>(matmul->input_value(1),
                                               ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0}));
            matmul->input(1).replace_source_output(transpose);
        }
    }

    return mat_mul;
}

ov::pass::AUGRUCellFusion::AUGRUCellFusion() {
    MATCHER_SCOPE(AUGRUCellFusion);

    // we can't determine hidden_size or input_size in this case
    const auto is_first_dim_static = [](const Output<Node>& output) -> bool {
        const auto& p_shape = output.get_partial_shape();
        return !(p_shape.rank().is_dynamic() || p_shape[1].is_dynamic());
    };

    auto concat_1 = wrap_type<ov::op::v0::Concat>({any_input(is_first_dim_static), any_input(is_first_dim_static)});
    auto matmul_1 = wrap_type<ov::op::v0::MatMul>({concat_1, any_input(is_first_dim_static)});
    auto add_1 = wrap_type<ov::op::v1::Add>({matmul_1, any_input()});
    // only Sigmoid is supported in the current version of AUGRUCell
    auto sigmoid = wrap_type<ov::op::v0::Sigmoid>({add_1});
    auto split = wrap_type<ov::op::v1::Split>({sigmoid, any_input()});
    split->set_output_size(2);
    // Split outputs can be at either index - create OutputVector for reuse
    OutputVector split_outputs{split->output(0), split->output(1)};
    auto split_out_any = make_shared<pattern::op::Or>(split_outputs);
    auto split_out_any2 = make_shared<pattern::op::Or>(split_outputs);
    auto multiply = wrap_type<ov::op::v1::Multiply>({split_out_any, any_input()});

    auto concat_2 = wrap_type<ov::op::v0::Concat>({any_input(), multiply});
    auto matmul_2 = wrap_type<ov::op::v0::MatMul>({concat_2, any_input(is_first_dim_static)});
    auto add_2 = wrap_type<ov::op::v1::Add>({matmul_2, any_input()});
    // only Tanh is supported in the current version of AUGRUCell
    auto tanh = wrap_type<ov::op::v0::Tanh>({add_2});

    auto subtract_1 = wrap_type<ov::op::v1::Subtract>({any_input(), any_input()});
    auto multiply_2 = wrap_type<ov::op::v1::Multiply>({subtract_1, split_out_any2});
    auto subtract_2 = wrap_type<ov::op::v1::Subtract>({any_input(), multiply_2});
    auto multiply_3 = wrap_type<ov::op::v1::Multiply>({subtract_2, tanh});

    auto multiply_4 = wrap_type<ov::op::v1::Multiply>({multiply_2, any_input()});
    auto add_3 = wrap_type<ov::op::v1::Add>({multiply_4, multiply_3});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry rg;
        auto pattern_map = m.get_pattern_map();
        auto concat = pattern_map.at(concat_1);
        auto X = concat->input_value(0);
        auto H = concat->input_value(1);

        auto h_pshape = H.get_partial_shape();
        auto x_pshape = X.get_partial_shape();

        auto hidden_size = h_pshape[1].get_length();
        auto input_size = x_pshape[1].get_length();

        auto axis_0 = rg.make<ov::op::v0::Constant>(i64, Shape{}, 0);
        auto axis_1 = rg.make<ov::op::v0::Constant>(i64, Shape{}, 1);

        auto A = pattern_map.at(subtract_1)->input_value(1);
        // biases are required
        auto bias_add_1 = get_bias_add(pattern_map.at(add_1), rg);
        auto split_bias_r_z = rg.make<ov::op::v1::Split>(bias_add_1->input_value(1), axis_1, 2);
        auto bias_add_2 = get_bias_add(pattern_map.at(add_2), rg);

        auto B = rg.make<ov::op::v0::Concat>(
            OutputVector{split_bias_r_z->output(1), split_bias_r_z->output(0), bias_add_2->input_value(1)},
            1);

        auto WRrz = get_weights_matmul(pattern_map.at(matmul_1), rg)->input_value(1);
        auto WRh = get_weights_matmul(pattern_map.at(matmul_2), rg)->input_value(1);

        auto split_lenghts = rg.make<ov::op::v0::Constant>(i64, Shape{2}, vector<int64_t>{input_size, hidden_size});
        auto split_WRrz = rg.make<ov::op::v1::VariadicSplit>(WRrz, axis_1, split_lenghts);
        auto split_W_r_z = rg.make<ov::op::v1::Split>(split_WRrz->output(0), axis_0, 2);
        auto split_R_r_z = rg.make<ov::op::v1::Split>(split_WRrz->output(1), axis_0, 2);
        auto split_WRh = rg.make<ov::op::v1::VariadicSplit>(WRh, axis_1, split_lenghts);
        auto Wzrh = rg.make<ov::op::v0::Concat>(
            OutputVector{split_W_r_z->output(1), split_W_r_z->output(0), split_WRh->output(0)},
            0);
        auto Rzrh = rg.make<ov::op::v0::Concat>(
            OutputVector{split_R_r_z->output(1), split_R_r_z->output(0), split_WRh->output(1)},
            0);

        auto squeeze_B = rg.make<ov::op::v0::Squeeze>(B, axis_0);
        auto cell =
            rg.make<op::internal::AUGRUCell>(X, H, Wzrh, Rzrh, squeeze_B, A, H.get_partial_shape()[1].get_length());

        cell->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info(m.get_matched_nodes(), rg.get());
        replace_node(m.get_match_root(), cell);
        return true;
    };

    auto m = make_shared<Matcher>(add_3, matcher_name);
    this->register_matcher(m, callback);
}
