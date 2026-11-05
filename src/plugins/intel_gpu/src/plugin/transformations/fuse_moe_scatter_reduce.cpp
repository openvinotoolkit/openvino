// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_scatter_reduce.hpp"

#include <memory>

#include "intel_gpu/op/moe_scatter_reduce.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;
using namespace ov::op;

FuseMoEScatterReduce::FuseMoEScatterReduce() {
    auto row_lut_m = any_input(rank_equals(1) && type_matches_any({ov::element::i32, ov::element::i64}));
    auto data_m = any_input(rank_equals(2) && has_static_dim(1));
    auto rows_m = wrap_type<v7::Gather, v8::Gather>({data_m, row_lut_m, 0}, consumers_count(1), {{"batch_dims", 0}});

    auto weights_m = any_input(rank_equals(2) && has_static_dim(1));
    auto flat_weights_m = wrap_type<v1::Reshape>({weights_m, any_input()}, rank_equals(1) && consumers_count(1));
    auto col_weights_m = wrap_type<v0::Unsqueeze>({flat_weights_m, any_input()}, shape_matches("?, 1") && consumers_count(1));
    auto weighted_m = wrap_type<v1::Multiply>({rows_m, col_weights_m}, consumers_count(1));
    auto split_m = wrap_type<v1::Reshape>({weighted_m, any_input()}, rank_equals(3) && consumers_count(1));
    auto reduce_m = wrap_type<v1::ReduceSum>({split_m, wrap_type<v0::Constant>()}, {{"keep_dims", false}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto reduce = ov::as_type_ptr<v1::ReduceSum>(m.get_match_root());
        if (!reduce || transformation_callback(reduce)) {
            return false;
        }
        const auto& pm = m.get_pattern_value_map();
        const auto data = pm.at(data_m);
        const auto weights = pm.at(weights_m);
        const auto row_lut = pm.at(row_lut_m);
        if (!reduce->reduction_axes_constant() || reduce->get_reduction_axes() != ov::AxisSet{1}) {
            return false;
        }

        const auto& weights_shape = weights.get_partial_shape();
        const auto top_k = weights_shape[1];
        const auto hidden = data.get_partial_shape()[1];
        const auto& split_shape = pm.at(split_m).get_partial_shape();
        if (split_shape[1] != top_k || split_shape[2] != hidden) {
            return false;
        }
        // The Multiply gives the T * K weights to the gathered rows, except for K = 1, where it can broadcast one weight to all the rows
        const auto& rows = row_lut.get_partial_shape()[0];
        const bool same_num_rows = (weights_shape[0].is_static() && weights_shape[0] == rows) ||
                                   ov::symbol::are_equal(weights_shape[0].get_symbol(), rows.get_symbol());
        if (top_k == 1 && !same_num_rows) {
            return false;
        }

        auto combine = std::make_shared<ov::intel_gpu::op::MoEScatterReduce>(data, weights, row_lut);
        combine->set_friendly_name(reduce->get_friendly_name());
        ov::copy_runtime_info({pm.at(rows_m).get_node_shared_ptr(),
                               pm.at(weighted_m).get_node_shared_ptr(),
                               pm.at(split_m).get_node_shared_ptr(),
                               reduce},
                              combine);
        ov::replace_node(reduce, combine);
        return true;
    };

    auto m = std::make_shared<Matcher>(reduce_m, "FuseMoEScatterReduce");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
