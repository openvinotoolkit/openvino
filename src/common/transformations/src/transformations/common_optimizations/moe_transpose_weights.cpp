// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moe_transpose_weights.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass;

using ov::pass::pattern::any_input;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
ov::pass::VectorizedMOE2GEMMTransposeWeights::VectorizedMOE2GEMMTransposeWeights() {
    MATCHER_SCOPE(VectorizedMOE2GEMMTransposeWeights);

    auto experts_input = wrap_type<v1::Reshape>({any_input(), any_input()});
    auto tile = wrap_type<v0::Tile>({experts_input, any_input()});
    auto after_tile_reshape = wrap_type<v1::Reshape>({tile, any_input()});
    auto gate_up_matmul =
        wrap_type<v0::MatMul>({after_tile_reshape, any_input()}, {{"transpose_a", false}, {"transpose_b", false}});
    auto gate_up_add = wrap_type<v1::Add>({gate_up_matmul, any_input()});

    auto slice1 = wrap_type<v8::Slice>({gate_up_add, any_input(), any_input(), any_input(), any_input()});
    auto clamp = wrap_type<v0::Clamp>({slice1});
    auto add1 = wrap_type<v1::Add>({clamp, ov::pass::pattern::wrap_const()});

    auto slice2 = wrap_type<v8::Slice>({gate_up_add, any_input(), any_input(), any_input(), any_input()});
    auto minimum1 = wrap_type<v1::Minimum>({slice2, ov::pass::pattern::wrap_const()});
    auto swish_beta = ov::pass::pattern::wrap_const();
    auto swish = wrap_type<ov::op::v4::Swish>({minimum1, swish_beta});

    auto multiply2 = wrap_type<v1::Multiply>({add1, swish});

    auto down_proj_matmul =
        wrap_type<v0::MatMul>({multiply2, any_input()}, {{"transpose_a", false}, {"transpose_b", false}});
    auto down_proj_add = wrap_type<v1::Add>({down_proj_matmul, ov::pass::pattern::wrap_const()});
    auto end_reshape = wrap_type<v1::Reshape>({down_proj_add, any_input()});

    auto router_topk_indices = any_input();
    auto scatter_elements_update =
        wrap_type<ov::op::v12::ScatterElementsUpdate>({any_input(), router_topk_indices, any_input(), any_input()});

    auto router_transpose = wrap_type<v1::Transpose>({scatter_elements_update, any_input()});
    auto router_reshape = wrap_type<v1::Reshape>({router_transpose, any_input()});
    auto unsqueeze_routing_weights = wrap_type<v0::Unsqueeze>({router_reshape, any_input()});

    auto mul3 = wrap_type<v1::Multiply>({end_reshape, unsqueeze_routing_weights});
    auto reduce_sum = wrap_type<v1::ReduceSum>({mul3, any_input()}, {{"keep_dims", false}});
    auto moe_pattern = reduce_sum;

    matcher_pass_callback callback = [=](Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        auto gate_matmul = ov::as_type_ptr<v0::MatMul>(pm.at(gate_up_matmul).get_node_shared_ptr());
        auto down_matmul = ov::as_type_ptr<v0::MatMul>(pm.at(down_proj_matmul).get_node_shared_ptr());
        if (!gate_matmul || !down_matmul)
            return false;

        auto insert_transpose = [&](const std::shared_ptr<v0::MatMul>& matmul) -> bool {
            auto weight_output = matmul->input_value(1);
            auto transpose_input = weight_output;
            std::shared_ptr<v0::Convert> decompress_convert;

            if (auto convert = ov::as_type_ptr<v0::Convert>(weight_output.get_node_shared_ptr())) {
                if (ov::is_decompression(convert) && convert->output(0).get_target_inputs().size() == 1) {
                    decompress_convert = convert;
                    transpose_input = convert->input_value(0);
                }
            }

            const auto& pshape = weight_output.get_partial_shape();
            const auto rank = pshape.rank();
            if (!rank.is_static() || rank.get_length() < 2)
                return false;

            std::vector<int64_t> transpose_order(rank.get_length());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::reverse(transpose_order.end() - 2, transpose_order.end());

            auto order_const = v0::Constant::create(ov::element::i64, {transpose_order.size()}, transpose_order);
            auto transpose = std::make_shared<v1::Transpose>(transpose_input, order_const);

            if (ov::is_type<v0::Constant>(transpose_input.get_node_shared_ptr())) {
                transpose->get_rt_info()["postponed_constant"] = true;
                ov::pass::disable_constant_folding(transpose);
            }

            ov::NodeVector rt_sources{transpose_input.get_node_shared_ptr()};
            if (auto weight_node = weight_output.get_node_shared_ptr()) {
                rt_sources.push_back(weight_node);
            }
            ov::copy_runtime_info(rt_sources, transpose);
            register_new_node(transpose);

            transpose->validate_and_infer_types();

            Output<Node> matmul_weight_input = transpose;

            if (decompress_convert) {
                decompress_convert->input(0).replace_source_output(transpose);
                decompress_convert->validate_and_infer_types();
                ov::mark_as_decompression(decompress_convert);
                matmul_weight_input = decompress_convert;
            }

            matmul->input(1).replace_source_output(matmul_weight_input);
            matmul->set_transpose_b(true);
            matmul->validate_and_infer_types();

            return true;
        };

        if (!insert_transpose(gate_matmul) || !insert_transpose(down_matmul))
            return false;

        ov::copy_runtime_info(m.get_matched_nodes(), {gate_matmul, down_matmul});
        return true;
    };

    auto matcher = std::make_shared<Matcher>(moe_pattern, matcher_name);
    this->register_matcher(matcher, callback);
}
