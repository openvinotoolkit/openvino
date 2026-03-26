// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_3gemm_compressed.hpp"

#include <memory>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
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
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu {
FuseMOE3GemmCompressed::FuseMOE3GemmCompressed() {
    using namespace ov::pass::pattern;
    using namespace ov::pass;
#define ANY any_input()

    auto hidden_state_m = ANY;
    auto hidden_state_reshape = optional<ov::op::v1::Reshape>({hidden_state_m, ANY});
    auto matmul = wrap_type<ov::op::v0::MatMul>({hidden_state_reshape, ANY}, consumers_count(1));

    // ── Softmax routing branch ──────────────────────────────────────────
    auto sm_softmax = wrap_type<ov::op::v8::Softmax>({matmul}, consumers_count(1));
    auto sm_topk = wrap_type<ov::op::v11::TopK>({sm_softmax, ANY});
    sm_topk->set_output_size(2);

    auto sm_reduce = wrap_type<ov::op::v1::ReduceSum>({sm_topk->output(0), ANY}, consumers_count(1));
    auto sm_norm = wrap_type<ov::op::v1::Divide>({sm_topk->output(0), sm_reduce}, consumers_count(1));
    auto sm_convert_topk = optional<ov::op::v0::Convert>({sm_topk->output(1)});
    auto sm_transpose = wrap_type<ov::op::v1::Transpose>({sm_norm, ANY}, consumers_count(1));
    auto sm_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({sm_transpose, ANY}, consumers_count(1));

    // ── Sigmoid+bias routing branch ─────────────────────────────────────
    auto sig_sigmoid = wrap_type<ov::op::v0::Sigmoid>({matmul});
    auto sig_routing_bias = ANY;
    auto sig_add = wrap_type<ov::op::v1::Add>({sig_sigmoid, sig_routing_bias}, consumers_count(1));
    auto sig_topk = wrap_type<ov::op::v11::TopK>({sig_add, ANY});
    sig_topk->set_output_size(2);

    auto sig_convert_topk = optional<ov::op::v0::Convert>({sig_topk->output(1)});
    auto sig_gather_el = wrap_type<ov::op::v6::GatherElements>({sig_sigmoid, sig_convert_topk});
    auto sig_reduce = wrap_type<ov::op::v1::ReduceSum>({sig_gather_el, ANY}, consumers_count(1));

    // Note: only scalar eps is supported for now
    auto sig_eps_value = wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
        return ov::shape_size(output.get_shape()) == 1;
    });
    auto sig_add_eps = wrap_type<ov::op::v1::Add>({sig_reduce, sig_eps_value}, consumers_count(1));
    auto sig_norm = wrap_type<ov::op::v1::Divide>({sig_gather_el, sig_add_eps}, consumers_count(1));
    auto sig_slice = optional<ov::op::v8::Slice>({sig_norm, ANY, ANY, ANY, ANY});
    auto sig_transpose = wrap_type<ov::op::v1::Transpose>({sig_slice, ANY}, consumers_count(1));
    auto sig_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({sig_transpose, ANY}, consumers_count(1));

    // ── Or-pattern: combine both branches ───────────────────────────────
    auto topk_idces = sm_convert_topk | sig_convert_topk;
    auto unsqueeze_moe = sm_unsqueeze | sig_unsqueeze;

    // ── Common: hidden state + compressed weights + MOECompressed ───────
    auto gate_wei_m = wrap_const();
    auto gate_scale_m = ANY;
    auto gate_zp_m = ANY;
    auto up_wei_m = wrap_const();
    auto up_scale_m = ANY;
    auto up_zp_m = ANY;
    auto down_wei_m = wrap_const();
    auto down_scale_m = ANY;
    auto down_zp_m = ANY;

    ov::OutputVector moe_inputs = {hidden_state_reshape | hidden_state_m,
                                   unsqueeze_moe,
                                   topk_idces,
                                   gate_wei_m,
                                   gate_scale_m,
                                   gate_zp_m,
                                   up_wei_m,
                                   up_scale_m,
                                   up_zp_m,
                                   down_wei_m,
                                   down_scale_m,
                                   down_zp_m};
    auto moe_compressed_no_shared_m = wrap_type<ov::op::internal::MOECompressed>(moe_inputs);

    // ── Shared expert weights ───────────────────────────────────────────
    auto shared_gate_wei_m = wrap_const();
    auto shared_gate_scale_m = ANY;
    auto shared_gate_zp_m = ANY;
    auto shared_up_wei_m = wrap_const();
    auto shared_up_scale_m = ANY;
    auto shared_up_zp_m = ANY;
    auto shared_down_wei_m = wrap_const();
    auto shared_down_scale_m = ANY;
    auto shared_down_zp_m = ANY;

    ov::OutputVector moe_inputs_shared = moe_inputs;
    moe_inputs_shared.push_back(shared_gate_wei_m);
    moe_inputs_shared.push_back(shared_gate_scale_m);
    moe_inputs_shared.push_back(shared_gate_zp_m);
    moe_inputs_shared.push_back(shared_up_wei_m);
    moe_inputs_shared.push_back(shared_up_scale_m);
    moe_inputs_shared.push_back(shared_up_zp_m);
    moe_inputs_shared.push_back(shared_down_wei_m);
    moe_inputs_shared.push_back(shared_down_scale_m);
    moe_inputs_shared.push_back(shared_down_zp_m);
    auto shared_gate_gate_wei_m = ANY;
    moe_inputs_shared.push_back(shared_gate_gate_wei_m);
    auto moe_compressed_shared_m = wrap_type<ov::op::internal::MOECompressed>(moe_inputs_shared);

    auto moe_compressed_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{moe_compressed_no_shared_m, moe_compressed_shared_m});
#undef ANY
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto moe_compressed = ov::as_type_ptr<ov::op::internal::MOECompressed>(pattern_map.at(moe_compressed_m).get_node_shared_ptr());
        if (!moe_compressed || transformation_callback(moe_compressed)) {
            return false;
        }

        auto config = moe_compressed->get_config();
        bool has_shared_expert = pattern_map.count(shared_gate_wei_m) > 0;
        if (!has_shared_expert) {
            config.num_shared_expert = 0;
        }
        // When Optional<Reshape> is absent, the optional pattern is NOT in the map.
        auto hs_reshaped = pattern_map.count(hidden_state_reshape)
            ? pattern_map.at(hidden_state_reshape)
            : pattern_map.at(hidden_state_m);
        OutputVector args{
            hs_reshaped,
            pattern_map.at(matmul),
            pattern_map.at(gate_wei_m),
            pattern_map.at(gate_scale_m),
            pattern_map.at(gate_zp_m),
            pattern_map.at(up_wei_m),
            pattern_map.at(up_scale_m),
            pattern_map.at(up_zp_m),
            pattern_map.at(down_wei_m),
            pattern_map.at(down_scale_m),
            pattern_map.at(down_zp_m),
        };
        if (pattern_map.count(sig_routing_bias)) {
            args.push_back(pattern_map.at(sig_routing_bias));
            args.push_back(pattern_map.at(sig_eps_value));
            config.routing_type = ov::op::internal::MOECompressed::RoutingType::SIGMOID_BIAS;
        } else if (has_shared_expert) {
            // SOFTMAX + shared expert: insert dummy placeholders at indices 11-12
            // so that shared expert inputs always start at index 13.
            auto dummy_bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.0f});
            auto dummy_eps = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.0f});
            args.push_back(dummy_bias);
            args.push_back(dummy_eps);
        }
        if (has_shared_expert) {
            args.push_back(pattern_map.at(shared_gate_wei_m));
            args.push_back(pattern_map.at(shared_gate_scale_m));
            args.push_back(pattern_map.at(shared_gate_zp_m));
            args.push_back(pattern_map.at(shared_up_wei_m));
            args.push_back(pattern_map.at(shared_up_scale_m));
            args.push_back(pattern_map.at(shared_up_zp_m));
            args.push_back(pattern_map.at(shared_down_wei_m));
            args.push_back(pattern_map.at(shared_down_scale_m));
            args.push_back(pattern_map.at(shared_down_zp_m));
            args.push_back(pattern_map.at(shared_gate_gate_wei_m));
        }

        std::shared_ptr<ov::Node> moe_router_fused = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, config);
        ov::copy_runtime_info(moe_compressed, moe_router_fused);

        // If MOECompressed's first input was the original (un-reshaped) hidden state
        // but the fused op works on the flattened 2D input, reshape the output back.
        if (moe_compressed->input_value(0) != hs_reshaped) {
            auto hidden_state_shape = std::make_shared<ov::op::v3::ShapeOf>(pattern_map.at(hidden_state_m));
            moe_router_fused = std::make_shared<ov::op::v1::Reshape>(moe_router_fused, hidden_state_shape, false);
            ov::copy_runtime_info(moe_compressed, {hidden_state_shape, moe_router_fused});
        }

        moe_router_fused->set_friendly_name(moe_compressed->get_friendly_name());
        ov::replace_node(moe_compressed, moe_router_fused);

        return true;
    };

    auto m = std::make_shared<Matcher>(moe_compressed_m, "FuseMOE3GemmCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
