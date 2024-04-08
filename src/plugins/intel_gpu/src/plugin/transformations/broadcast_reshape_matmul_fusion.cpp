// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_reshape_matmul_fusion.hpp"

#include "intel_gpu/op/gemm.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

BroadcastReshapeMatmulFusion::BroadcastReshapeMatmulFusion() {
    using namespace ov::pass::pattern;

    auto not_reshape = [](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Reshape>(output.get_node_shared_ptr()) == nullptr;
    };

    auto broadcast_rank_equals_and_has_static_dims = [](const ov::Output<ov::Node>& output) -> bool {
        return rank_equals(5)(output) && has_static_dims({2, 3}) && consumers_count(1);
    };

    auto reshape_rank_equals_and_has_static_dim = [](const ov::Output<ov::Node>& output) -> bool {
        return rank_equals(4)(output) && has_static_dim(2) && consumers_count(1);
    };

    auto input_a_m = any_input(not_reshape);
    auto input_b_m = any_input(not_reshape);

    auto broadcast_a_target_shape_m = wrap_type<ov::op::v0::Constant>();
    auto broadcast_a_m = wrap_type<ov::op::v3::Broadcast>({input_a_m, broadcast_a_target_shape_m}, broadcast_rank_equals_and_has_static_dims);
    auto broadcast_b_target_shape_m = wrap_type<ov::op::v0::Constant>();
    auto broadcast_b_m = wrap_type<ov::op::v3::Broadcast>({input_b_m, broadcast_b_target_shape_m}, broadcast_rank_equals_and_has_static_dims);

    auto reshape_a_pattern_m = wrap_type<ov::op::v0::Constant>();
    auto reshape_a_m = wrap_type<ov::op::v1::Reshape>({broadcast_a_m, reshape_a_pattern_m}, reshape_rank_equals_and_has_static_dim);
    auto reshape_b_pattern_m = wrap_type<ov::op::v0::Constant>();
    auto reshape_b_m = wrap_type<ov::op::v1::Reshape>({broadcast_b_m, reshape_b_pattern_m}, reshape_rank_equals_and_has_static_dim);

    auto matmul_in_a = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{input_a_m, reshape_a_m});
    auto matmul_in_b = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{input_b_m, reshape_b_m});

    auto matmul_m = wrap_type<op::Gemm>({matmul_in_a, matmul_in_b});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul = std::dynamic_pointer_cast<op::Gemm>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(m.get_match_root())) {
            return false;
        }

        auto target_shape_a = std::vector<int32_t>();
        auto target_shape_b = std::vector<int32_t>();
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();
        auto order_a = matmul->get_input0_transpose_order();
        auto order_b = matmul->get_input1_transpose_order();

        auto valid_transpose_order = [](const std::vector<int64_t>& order) {
            return order.size() == 4 && order[1] == 2;
        };

        auto valid_broadcast_target_shape = [](const std::vector<int32_t>& target_shape) {
            return std::count_if(target_shape.begin(), target_shape.end(), [](int32_t s) { return s != 1; }) == 1;
        };

        if (pattern_map.count(broadcast_a_m) > 0) {
            if (!valid_transpose_order(order_a))
                return false;
            auto broadcast_a = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(pattern_map.at(broadcast_a_m).get_node_shared_ptr());
            if (!broadcast_a || broadcast_a->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL)
                return false;
            auto broadcast_a_target_shape = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(broadcast_a_target_shape_m).get_node_shared_ptr());
            target_shape_a = broadcast_a_target_shape->cast_vector<int32_t>();
            if (!valid_broadcast_target_shape(target_shape_a))
                return false;
            input_a_output_idx = broadcast_a->get_input_source_output(0).get_index();
        }
        if (pattern_map.count(broadcast_b_m) > 0) {
            if (!valid_transpose_order(order_b))
                return false;
            auto broadcast_b = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(pattern_map.at(broadcast_b_m).get_node_shared_ptr());
            if (!broadcast_b || broadcast_b->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL)
                return false;
            auto broadcast_b_target_shape = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(broadcast_b_target_shape_m).get_node_shared_ptr());
            target_shape_b = broadcast_b_target_shape->cast_vector<int32_t>();
            if (!valid_broadcast_target_shape(target_shape_b))
                return false;
            input_b_output_idx = broadcast_b->get_input_source_output(0).get_index();
        }

        auto pattern_a = std::vector<int64_t>();
        auto pattern_b = std::vector<int64_t>();

        auto valid_reshape_pattern = [](const std::vector<int64_t>& pattern) {
            return std::count_if(pattern.begin(), pattern.end(), [](int64_t p) { return p == -1; }) == 0;
        };

        if (pattern_map.count(reshape_a_m) > 0) {
            auto reshape_a_pattern = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(reshape_a_pattern_m).get_node_shared_ptr());
            pattern_a = reshape_a_pattern->cast_vector<int64_t>();
            if (!valid_reshape_pattern(pattern_a))
                return false;
        }
        if (pattern_map.count(reshape_b_m) > 0) {
            auto reshape_b_pattern = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(reshape_b_pattern_m).get_node_shared_ptr());
            pattern_b = reshape_b_pattern->cast_vector<int64_t>();
            if (!valid_reshape_pattern(pattern_b))
                return false;
        }

        auto input_a = ov::Output<Node>(pattern_map.at(input_a_m).get_node_shared_ptr(), input_a_output_idx);
        auto input_b = ov::Output<Node>(pattern_map.at(input_b_m).get_node_shared_ptr(), input_b_output_idx);
        auto order_c = matmul->get_output_transpose_order();

        auto gemm = std::make_shared<op::Gemm>(input_a,
                                               input_b,
                                               target_shape_a,
                                               target_shape_b,
                                               pattern_a,
                                               pattern_b,
                                               order_a,
                                               order_b,
                                               order_c);
        gemm->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(matmul, gemm);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, "BroadcastReshapeMatmulFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
