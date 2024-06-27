// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unsqueeze_broadcast_reshape_matmul_fusion.hpp"

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/kv_cache.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

UnsqueezeBroadcastReshapeMatmulFusion::UnsqueezeBroadcastReshapeMatmulFusion() {
    using namespace ov::pass::pattern;

    auto not_reshape = [](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Reshape>(output.get_node_shared_ptr()) == nullptr;
    };

    auto unsqueeze_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        return rank_equals(5)(output) && consumers_count(1);
    };

    auto broadcast_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        const auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(output.get_node_shared_ptr());
        if (!broadcast || broadcast->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL)
            return false;
        return rank_equals(5)(output) && consumers_count(1);
    };

    auto reshape_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        return rank_equals(4)(output) && consumers_count(1);
    };

    auto input_a_m = any_input(not_reshape);
    auto input_b_m = wrap_type<ov::intel_gpu::op::KVCache>({any_input(), any_input()});
    auto axes_const_m = wrap_type<ov::op::v0::Constant>();
    auto unsqueeze_m = wrap_type<ov::op::v0::Unsqueeze>({input_b_m, axes_const_m}, unsqueeze_predicate);
    auto broadcast_m = wrap_type<ov::op::v3::Broadcast>({unsqueeze_m, any_input()}, broadcast_predicate);
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({broadcast_m, any_input()}, reshape_predicate);
    auto matmul_m = wrap_type<op::Gemm>({input_a_m, reshape_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();

        auto valid_broadcast_target_shape = [](const std::vector<int32_t>& target_shape) {
            return std::count_if(target_shape.begin(), target_shape.end(), [](int32_t s) { return s != 1; }) == 1;
        };
        auto broadcast = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(pattern_map.at(broadcast_m).get_node_shared_ptr());
        auto target_shape_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(broadcast->get_input_node_shared_ptr(1));
        if (target_shape_constant) {
            auto target_shape_val = target_shape_constant->cast_vector<int32_t>();
            if (!valid_broadcast_target_shape(target_shape_val))
                return false;
        }

        auto input_a = pattern_map.at(input_a_m).get_node_shared_ptr();
        auto input_b = pattern_map.at(input_b_m).get_node_shared_ptr();

        auto matmul = std::dynamic_pointer_cast<op::Gemm>(m.get_match_root());
        auto order_a = matmul->get_input0_transpose_order();
        auto order_b = matmul->get_input1_transpose_order();
        auto order_c = matmul->get_output_transpose_order();

        auto gemm = std::make_shared<op::Gemm>(input_a,
                                               input_b,
                                               order_a,
                                               order_b,
                                               order_c);
        gemm->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(matmul, gemm);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, "UnsqueezeBroadcastReshapeMatmulFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
