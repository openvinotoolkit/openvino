// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_transpose_fusion.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"

using ov::pass::pattern::op::Or;

namespace ov::intel_gpu {

/// Compose two transpose orders: first apply `inner`, then `outer`.
/// compose_orders({a,b,c}, {x,y,z}) = {a[inner[x]], a[inner[y]], a[inner[z]]}
static std::vector<int64_t> compose_orders(const std::vector<int64_t>& inner,
                                           const std::vector<int64_t>& outer) {
    OPENVINO_ASSERT(inner.size() == outer.size());
    std::vector<int64_t> result(inner.size());
    for (size_t i = 0; i < inner.size(); ++i)
        result[i] = inner[outer[i]];
    return result;
}

/// Returns the {0,2,1,3} permutation of `transpose` if it is a rank-4
/// heads<->seq swap with a constant order, otherwise an empty vector.
static std::vector<int64_t> match_heads_seq_swap(const std::shared_ptr<ov::op::v1::Transpose>& transpose) {
    auto order_node =
        ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!order_node)
        return {};

    auto order = order_node->cast_vector<int64_t>();
    if (order != std::vector<int64_t>{0, 2, 1, 3})
        return {};

    return order;
}

static bool is_rank_4(const ov::PartialShape& pshape) {
    return pshape.rank().is_static() && pshape.rank().get_length() == 4;
}

SDPATransposeFusion::SDPATransposeFusion() {
    using namespace ov::op;
    using namespace ov::pass::pattern;

    // A single matcher handles both SDPA flavors (MatcherPass supports only one
    // registered matcher):
    //
    // 1. Internal op::SDPA (produced by TransposeSDPAMatcher) followed by
    //    Transpose({0,2,1,3}) — absorb the Transpose into output_transpose_order.
    //
    // 2. Framework v13::ScaledDotProductAttention followed by
    //    Transpose({0,2,1,3}). TransposeSDPAMatcher converts v13 SDPA to the
    //    internal op::SDPA, but it bails out entirely when any input transpose
    //    moves the head_size dim (e.g. K preceded by Transpose({0,1,3,2}) as in
    //    pi05). In that case the SDPA stays a framework op; convert it here with
    //    identity input orders (leaving any input transposes as explicit ops)
    //    and absorb the output Transpose into output_transpose_order.
    //
    // Note: the internal op::SDPA derives from v13::ScaledDotProductAttention,
    // so wrap_type<v13::...> would also match internal SDPA nodes and resetting
    // their input orders to identity would corrupt Q/K/V interpretation. The
    // Or alternatives are ordered so that internal SDPA nodes always take the
    // first (op::SDPA) branch.
    auto sdpa_m = wrap_type<ov::intel_gpu::op::SDPA>(consumers_count(1));
    auto sdpa_v13_m = wrap_type<v13::ScaledDotProductAttention>(consumers_count(1));
    auto any_sdpa_m = std::make_shared<Or>(OutputVector{sdpa_m, sdpa_v13_m});
    auto transpose_m = wrap_type<v1::Transpose>({any_sdpa_m, any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto transpose = ov::as_type_ptr<v1::Transpose>(
            pattern_map.at(transpose_m).get_node_shared_ptr());
        if (!transpose)
            return false;

        auto order = match_heads_seq_swap(transpose);
        if (order.empty())
            return false;

        std::shared_ptr<ov::Node> sdpa_node;
        std::shared_ptr<ov::intel_gpu::op::SDPA> new_sdpa;

        if (pattern_map.count(sdpa_m) > 0) {
            auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(
                pattern_map.at(sdpa_m).get_node_shared_ptr());
            if (!sdpa || transformation_callback(sdpa))
                return false;

            // Only match rank-4 output.
            if (!is_rank_4(sdpa->get_output_partial_shape(0)))
                return false;

            // Compose output_transpose_order with the Transpose permutation.
            // Only apply when the SDPA's current output order is identity —
            // backbone GQA SDPA ops have custom orders that must not be touched.
            auto cur_out_order = sdpa->get_output_transpose_order();
            for (size_t i = 0; i < cur_out_order.size(); ++i) {
                if (cur_out_order[i] != static_cast<int64_t>(i))
                    return false;
            }
            auto new_out_order = compose_orders(cur_out_order, order);

            // Build replacement SDPA preserving the original's compression state.
            if (sdpa->get_kv_compressed()) {
                new_sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(
                    sdpa->input_values(),
                    sdpa->get_causal(),
                    sdpa->get_input0_transpose_order(),
                    sdpa->get_input1_transpose_order(),
                    sdpa->get_input2_transpose_order(),
                    new_out_order,
                    sdpa->get_quantization_attrs(),
                    sdpa->get_output_type());
            } else {
                new_sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(
                    sdpa->input_values(),
                    sdpa->get_causal(),
                    sdpa->get_input0_transpose_order(),
                    sdpa->get_input1_transpose_order(),
                    sdpa->get_input2_transpose_order(),
                    new_out_order,
                    sdpa->get_output_type());
            }
            sdpa_node = sdpa;
        } else {
            auto sdpa = ov::as_type_ptr<v13::ScaledDotProductAttention>(
                pattern_map.at(sdpa_v13_m).get_node_shared_ptr());
            if (!sdpa || transformation_callback(sdpa))
                return false;

            // Defensive: never touch internal op::SDPA nodes here — they are
            // handled by the op::SDPA branch above.
            if (ov::as_type_ptr<ov::intel_gpu::op::SDPA>(sdpa))
                return false;

            // Only match rank-4 inputs/output.
            if (!is_rank_4(sdpa->get_output_partial_shape(0)))
                return false;
            for (size_t i = 0; i < 3; ++i) {
                if (!is_rank_4(sdpa->get_input_partial_shape(i)))
                    return false;
            }

            // Keep identity input orders and only absorb the output Transpose.
            //
            // We deliberately do NOT absorb the Q/K/V input Transposes here.
            // Doing so is numerically correct (verified bitwise-identical), but
            // it is a measured performance regression: the SDPA kernel then has
            // to read the inputs with a transposed (non-contiguous head_size)
            // stride pattern, which costs more than the standalone permute it
            // would remove. In particular pi05's K uses Transpose({0,1,3,2}),
            // which moves head_size off the innermost dim; materializing K
            // contiguously with a separate permute and letting the SDPA read it
            // contiguously is faster (~3% on pi05) than an in-kernel strided
            // read. This is the same reason TransposeSDPAMatcher guards against
            // head_size-moving input transposes.
            new_sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(
                sdpa->input_values(),
                sdpa->get_causal(),
                ov::intel_gpu::op::SDPA::default_order(4),
                ov::intel_gpu::op::SDPA::default_order(4),
                ov::intel_gpu::op::SDPA::default_order(4),
                compose_orders(ov::intel_gpu::op::SDPA::default_order(4), order));
            sdpa_node = sdpa;
        }

        new_sdpa->set_friendly_name(sdpa_node->get_friendly_name());
        ov::copy_runtime_info(ov::NodeVector{sdpa_node, transpose}, new_sdpa);

        // Replace the SDPA node, then rewire Transpose consumers to the new
        // SDPA output (bypassing the now-dead Transpose).
        ov::replace_node(sdpa_node, new_sdpa);
        ov::replace_output_update_name(transpose->output(0),
                                       transpose->input_value(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_m,
                                                          "SDPATransposeFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
