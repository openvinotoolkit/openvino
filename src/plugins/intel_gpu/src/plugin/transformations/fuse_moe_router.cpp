// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_router.hpp"

#include <memory>

#include "intel_gpu/op/moe_router_fused.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/op.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;
using namespace ov::pass;
#define ANY any_input()

FuseMoESoftmaxRouter::FuseMoESoftmaxRouter() {
    auto routing_matmul = wrap_type<ov::op::v0::MatMul>();

    auto softmax_m = wrap_type<ov::op::v8::Softmax>({routing_matmul}, consumers_count(1));
    auto topk_k_m = wrap_const();
    auto topk_m = wrap_type<ov::op::v11::TopK>({softmax_m, topk_k_m});
    topk_m->set_output_size(2);

    auto reduce_m = wrap_type<ov::op::v1::ReduceSum>({topk_m->output(0), ANY}, consumers_count(1));
    auto norm_m = wrap_type<ov::op::v1::Divide>({topk_m->output(0), reduce_m});
    auto convert_topk_m = optional<ov::op::v0::Convert>({topk_m->output(1)});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& topk_in_shape = pattern_map.at(softmax_m).get_partial_shape();
        const auto& topk_out_shape = pattern_map.at(topk_m).get_partial_shape();
        for (const auto& shape : {topk_in_shape, topk_out_shape}) {
            if (shape.rank().is_dynamic() || shape.size() == 0 || shape.rbegin()->is_dynamic()) {
                return false;
            }
        }

        const OutputVector router_args{pattern_map.at(routing_matmul)};

        const size_t num_expert = topk_in_shape.rbegin()->get_length();
        const size_t top_k = topk_out_shape.rbegin()->get_length();
        const ov::intel_gpu::op::MoERouterFused::Config router_config{num_expert, top_k, ov::intel_gpu::op::MoERouterFused::RoutingType::SOFTMAX};

        auto router_node = std::make_shared<ov::intel_gpu::op::MoERouterFused>(router_args, router_config);
        ov::copy_runtime_info(m.get_match_root(), router_node);

        OPENVINO_ASSERT(ov::replace_output_update_name(pattern_map.at(norm_m), router_node->output(0)), "MoERouter fusion failed");
        const auto topk_node_m = pattern_map.at(topk_m).get_node_shared_ptr();
        OPENVINO_ASSERT(ov::replace_output_update_name(topk_node_m->output(1), router_node->output(1)), "MoERouter fusion failed");
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(norm_m, "FuseMoESoftmaxRouter");
    this->register_matcher(matcher, callback);
}

FuseMoESigmoidRouter::FuseMoESigmoidRouter() {
    auto routing_matmul = wrap_type<ov::op::v0::MatMul>();

    auto sigmoid_m = wrap_type<ov::op::v0::Sigmoid>({routing_matmul});
    auto routing_bias_m = ANY;
    auto add_m = wrap_type<ov::op::v1::Add>({sigmoid_m, routing_bias_m}, consumers_count(1));
    auto topk_k_m = wrap_const();
    auto topk_m = wrap_type<ov::op::v11::TopK>({add_m, topk_k_m});
    topk_m->set_output_size(2);

    auto convert_topk_m = optional<ov::op::v0::Convert>({topk_m->output(1)});
    auto gather_el_m = wrap_type<ov::op::v6::GatherElements>({sigmoid_m, convert_topk_m});
    auto reduce_m = wrap_type<ov::op::v1::ReduceSum>({gather_el_m, ANY}, consumers_count(1));

    // Note: only scalar eps is supported for now
    auto eps_value_m = wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
        return ov::shape_size(output.get_shape()) == 1;
    });
    auto add_eps_m = wrap_type<ov::op::v1::Add>({reduce_m, eps_value_m}, consumers_count(1));
    auto norm_m = wrap_type<ov::op::v1::Divide>({gather_el_m, add_eps_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& topk_in_shape = pattern_map.at(add_m).get_partial_shape();
        const auto& topk_out_shape = pattern_map.at(topk_m).get_partial_shape();
        for (const auto& shape : {topk_in_shape, topk_out_shape}) {
            if (shape.rank().is_dynamic() || shape.size() == 0 || shape.rbegin()->is_dynamic()) {
                return false;
            }
        }

        const size_t top_k = topk_out_shape.rbegin()->get_length();
        const size_t num_expert = topk_in_shape.rbegin()->get_length();
        const ov::intel_gpu::op::MoERouterFused::Config router_config{num_expert, top_k, ov::intel_gpu::op::MoERouterFused::RoutingType::SIGMOID_BIAS};

        OutputVector router_args{pattern_map.at(routing_matmul)};
        router_args.push_back(pattern_map.at(routing_bias_m));
        router_args.push_back(pattern_map.at(eps_value_m));

        auto router_node = std::make_shared<ov::intel_gpu::op::MoERouterFused>(router_args, router_config);
        ov::copy_runtime_info(m.get_match_root(), router_node);

        OPENVINO_ASSERT(ov::replace_output_update_name(pattern_map.at(norm_m), router_node->output(0)), "MoERouter fusion failed");
        OPENVINO_ASSERT(ov::replace_output_update_name(pattern_map.at(convert_topk_m), router_node->output(1)), "MoERouter fusion failed");

        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(norm_m, "FuseMoESigmoidRouter");
    this->register_matcher(matcher, callback);
}

FuseMoERouter::FuseMoERouter() {
    add_matcher<FuseMoESoftmaxRouter>();
    add_matcher<FuseMoESigmoidRouter>();
}

#undef ANY
}  // namespace ov::intel_gpu
