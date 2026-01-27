// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_3gemm_compressed.hpp"

#include <memory>

#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/slice.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

FuseMOE3GemmCompressed::FuseMOE3GemmCompressed() {
    using namespace ov::pass::pattern;

    // Inputs
    auto hidden_state = any_input();
    auto hidden_state_reshape = wrap_type<ov::op::v1::Reshape>({hidden_state, any_input()});
    auto routers = any_input();

    // Router subgraph
    auto router_mm = wrap_type<ov::op::v0::MatMul>({hidden_state_reshape, routers}, consumers_count(1));
    auto router_sm = wrap_type<ov::op::v8::Softmax>({router_mm}, consumers_count(1));
    auto topk = wrap_type<ov::op::v11::TopK>({router_sm, any_input()});
    topk->set_output_size(2);

    // Weights path: norm = topk_val / reduce_sum(topk_val)
    auto reduce_sum = wrap_type<ov::op::v1::ReduceSum>({topk->output(0), any_input()}, consumers_count(1));
    auto norm = wrap_type<ov::op::v1::Divide>({topk->output(0), reduce_sum->output(0)}, consumers_count(1));

    auto topk_idx_convert_m = wrap_type<ov::op::v0::Convert>({topk->output(1)});

    // Indices path: shape_of -> gather -> unsqueezes
    auto shape_of = wrap_type<ov::op::v3::ShapeOf>({topk_idx_convert_m}, consumers_count(1));
    auto norm_slice = wrap_type<ov::op::v8::Slice>({norm->output(0), any_input(), shape_of, any_input(), any_input()});

    // auto gather = wrap_type<ov::op::v8::Gather>({shape_of, any_input(), any_input()}, consumers_count(1));
    // auto unsq_idx = wrap_type<ov::op::v0::Unsqueeze>({gather, any_input()});

    // // Concat/broadcast path with independent const unsqueeze nodes
    // auto unsq_c0 = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});
    // auto unsq_c1 = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});

    // auto concat0 = wrap_type<ov::op::v0::Concat>({unsq_idx, unsq_c0}, consumers_count(1));
    // auto concat1 = wrap_type<ov::op::v0::Concat>({unsq_c1, unsq_idx, any_input()}, consumers_count(1));

    // auto bc = wrap_type<ov::op::v3::Broadcast>({any_input(), concat0}, consumers_count(1));
    auto scatter = wrap_type<ov::op::v12::ScatterElementsUpdate>(
        {any_input(), topk_idx_convert_m, norm_slice, any_input()},
        consumers_count(1));

    auto trans = wrap_type<ov::op::v1::Transpose>({scatter, any_input()}, consumers_count(1));
    auto reshape = wrap_type<ov::op::v1::Reshape>({trans, any_input()}, consumers_count(1));
    auto unsq_moe = wrap_type<ov::op::v0::Unsqueeze>({reshape, any_input()}, consumers_count(1));

    auto hidden_state_convert = optional<ov::op::v0::Convert>({hidden_state});

    // MOECompressed inputs
    auto gate_w  = wrap_type<ov::op::v0::Constant>();
    auto gate_sc = any_input();
    auto gate_zp = any_input();
    auto up_w    = wrap_type<ov::op::v0::Constant>();
    auto up_sc   = any_input();
    auto up_zp   = any_input();
    auto down_w  = wrap_type<ov::op::v0::Constant>();
    auto down_sc = any_input();
    auto down_zp = any_input();

    auto moe = wrap_type<ov::intel_gpu::op::MOECompressed>({
        hidden_state_convert->output(0),
        unsq_moe->output(0),
        topk_idx_convert_m,
        gate_w->output(0), gate_sc->output(0), gate_zp->output(0),
        up_w->output(0),   up_sc->output(0),   up_zp->output(0),
        down_w->output(0), down_sc->output(0), down_zp->output(0)
    });

    ov::matcher_pass_callback cb = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        std::cout << "moe_3gemm_fused_compressed matched" << std::endl;
        auto moe_node = ov::as_type_ptr<ov::intel_gpu::op::MOECompressed>(pm.at(moe).get_node_shared_ptr());
        if (!moe_node || transformation_callback(moe_node))
            return false;

        OutputVector args(11);
        auto hidden_state_reshape_node = pm.at(hidden_state_reshape).get_node_shared_ptr();
        if (hidden_state_reshape_node->get_element_type() == ov::element::f32) {
            auto hidden_state_convert = std::make_shared<ov::op::v0::Convert>(hidden_state_reshape_node, ov::element::f16);
            args[0] = hidden_state_convert;
        } else {
            args[0] = hidden_state_reshape_node;
        }

        auto routers_node = pm.at(router_mm).get_node_shared_ptr();
        if (routers_node->get_element_type() == ov::element::f32) {
            auto routers_convert = std::make_shared<ov::op::v0::Convert>(routers_node, ov::element::f16);
            args[1]  = routers_convert;
        } else {
            args[1]  = routers_node;
        }
        args[2]  = pm.at(gate_w);
        args[3]  = pm.at(gate_sc);
        args[4]  = pm.at(gate_zp);
        args[5]  = pm.at(up_w);
        args[6]  = pm.at(up_sc);
        args[7]  = pm.at(up_zp);
        args[8]  = pm.at(down_w);
        args[9]  = pm.at(down_sc);
        args[10] = pm.at(down_zp);

        auto fused = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, moe_node->get_config());
        // auto hs_node = pm.at(hidden_state).get_node_shared_ptr();
        // auto hs_shape = hs_node->get_output_partial_shape(0);

        // auto reshape_back = std::make_shared<ov::op::v1::Reshape>(fused, hs_shape);

        fused->set_friendly_name(moe_node->get_friendly_name());
        ov::copy_runtime_info(moe_node, fused);
        ov::replace_node(moe_node, fused);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(moe, "FuseMOE3GemmCompressed");
    register_matcher(matcher, cb);
}

}  // namespace ov::intel_gpu
