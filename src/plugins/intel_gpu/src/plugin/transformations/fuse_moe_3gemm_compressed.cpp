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
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/op.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {
using namespace ov::pass;
using namespace ov::pass::pattern;

FuseMOE3GemmCompressed::FuseMOE3GemmCompressed() {
    auto hidden_state_m = any_input();
    auto routers_m = any_input();
    auto hidden_state_reshape_m = wrap_type<ov::op::v1::Reshape>({hidden_state_m, any_input()});
    auto router_matmul_m = wrap_type<ov::op::v0::MatMul>({hidden_state_reshape_m, routers_m}, consumers_count(1));
    auto softmax_m = wrap_type<ov::op::v8::Softmax>({router_matmul_m}, consumers_count(1));
    auto topk_m = wrap_type<ov::op::v11::TopK>({softmax_m, any_input()});
    topk_m->set_output_size(2);
    // idx output
    auto topk_idx_m = optional<ov::op::v0::Convert>({topk_m->output(1)});

    // weight output
    auto reduce_sum_m = wrap_type<ov::op::v1::ReduceSum>({topk_m->output(0), any_input()}, consumers_count(1));
    auto norm_m = wrap_type<ov::op::v1::Divide>({topk_m->output(0), reduce_sum_m->output(0)}, consumers_count(1));

    auto bc_m = wrap_type<ov::op::v3::Broadcast>({any_input(), any_input()});
    auto shape_of_m = wrap_type<ov::op::v3::ShapeOf>({topk_idx_m}, consumers_count(1));
    auto norm_slice = optional<ov::op::v8::Slice>({norm_m, any_input(), shape_of_m, any_input(), any_input()}, consumers_count(1));

    auto scatter_m = wrap_type<ov::op::v12::ScatterElementsUpdate>({bc_m, topk_idx_m, norm_m | norm_slice, any_input()}, consumers_count(1));
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({scatter_m, any_input()}, consumers_count(1));
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({transpose_m, any_input()}, consumers_count(1));
    auto unsqueeze_moe_m = wrap_type<ov::op::v0::Unsqueeze>({reshape_m, any_input()}, consumers_count(1));

    auto gate_wei_m = wrap_type<ov::op::v0::Constant>();
    auto gate_scale_m = any_input();
    auto gate_zp_m = any_input();
    auto up_wei_m = wrap_type<ov::op::v0::Constant>();
    auto up_scale_m = any_input();
    auto up_zp_m = any_input();
    auto down_wei_m = wrap_type<ov::op::v0::Constant>();
    auto down_scale_m = any_input();
    auto down_zp_m = any_input();

    auto hidden_state_convert_m = optional<ov::op::v0::Convert>({hidden_state_m});

    // moe compressed
    auto moe_compressed_m = wrap_type<ov::intel_gpu::op::MOECompressed>({hidden_state_reshape_m | hidden_state_convert_m,
                                                                         unsqueeze_moe_m->output(0),
                                                                         topk_idx_m->output(0),
                                                                         gate_wei_m->output(0),
                                                                         gate_scale_m->output(0),
                                                                         gate_zp_m->output(0),
                                                                         up_wei_m->output(0),
                                                                         up_scale_m->output(0),
                                                                         up_zp_m->output(0),
                                                                         down_wei_m->output(0),
                                                                         down_scale_m->output(0),
                                                                         down_zp_m->output(0)});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto moe_compressed = ov::as_type_ptr<ov::intel_gpu::op::MOECompressed>(pattern_map.at(moe_compressed_m).get_node_shared_ptr());
        if (!moe_compressed || transformation_callback(moe_compressed)) {
            return false;
        }

        OutputVector args(11);
        auto hidden_state_reshape_node = pattern_map.at(hidden_state_reshape_m).get_node_shared_ptr();
        if (hidden_state_reshape_node->get_element_type() == ov::element::f32) {
            auto hidden_state_reshape_convert = std::make_shared<ov::op::v0::Convert>(hidden_state_reshape_node, ov::element::f16);
            args[0] = hidden_state_reshape_convert;
        } else {
            args[0] = hidden_state_reshape_node;
        }

        auto routers_node = pattern_map.at(router_matmul_m).get_node_shared_ptr();
        if (routers_node->get_element_type() == ov::element::f32) {
            auto routers_convert = std::make_shared<ov::op::v0::Convert>(routers_node, ov::element::f16);
            args[1]  = routers_convert;
        } else {
            args[1]  = routers_node;
        }
        args[2] = pattern_map.at(gate_wei_m);
        args[3] = pattern_map.at(gate_scale_m);
        args[4] = pattern_map.at(gate_zp_m);
        args[5] = pattern_map.at(up_wei_m);
        args[6] = pattern_map.at(up_scale_m);
        args[7] = pattern_map.at(up_zp_m);
        args[8] = pattern_map.at(down_wei_m);
        args[9] = pattern_map.at(down_scale_m);
        args[10] = pattern_map.at(down_zp_m);

        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, moe_compressed->get_config());

        bool need_reshape = false;
        std::shared_ptr<ov::Node> moe_reshape;

        const auto& convert = moe_compressed->get_users().front();
        if (ov::is_type<ov::op::v0::Convert>(convert)) {
            const auto& add = convert->get_users().front();
            if (ov::is_type<ov::op::v1::Add>(add)) {
                const auto reshape = add->get_input_node_ptr(0);
                if (ov::is_type<ov::op::v1::Reshape>(reshape)) {
                    auto target_shape = reshape->input_value(1);
                    need_reshape = true;
                    moe_reshape = std::make_shared<ov::op::v1::Reshape>(moe_3gemm_fused_compressed, target_shape, false);
                }
            }
        }

        if (need_reshape) {
            moe_reshape->set_friendly_name(moe_compressed->get_friendly_name());
            ov::copy_runtime_info(moe_compressed, moe_reshape);
            ov::replace_node(moe_compressed, moe_reshape);
        } else {
            moe_3gemm_fused_compressed->set_friendly_name(moe_compressed->get_friendly_name());
            ov::copy_runtime_info(moe_compressed, moe_3gemm_fused_compressed);
            ov::replace_node(moe_compressed, moe_3gemm_fused_compressed);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_compressed_m, "FuseMOE3GemmCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
