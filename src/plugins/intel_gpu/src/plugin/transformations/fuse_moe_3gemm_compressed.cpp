// Copyright (C) 2018-2025 Intel Corporation
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
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

FuseMOE3GemmCompressed::FuseMOE3GemmCompressed() {
    auto hidden_state_m = any_input();
    auto routers_m = any_input();
    auto router_matmul_m = wrap_type<ov::op::v0::MatMul>({hidden_state_m, routers_m}, consumers_count(1));
    auto softmax_m = wrap_type<ov::op::v8::Softmax>({router_matmul_m}, consumers_count(1));
    auto topk_m = wrap_type<ov::op::v11::TopK>({softmax_m, any_input()});
    topk_m->set_output_size(2);

    // weight output
    auto reduce_sum_m = wrap_type<ov::op::v1::ReduceSum>({topk_m->output(0), any_input()}, consumers_count(1));
    auto norm_m = wrap_type<ov::op::v1::Divide>({topk_m->output(0), reduce_sum_m->output(0)}, consumers_count(1));

    // idx output
    auto shape_of_m = wrap_type<ov::op::v3::ShapeOf>({topk_m->output(1)}, consumers_count(1));
    auto gather_m = wrap_type<ov::op::v8::Gather>({shape_of_m, any_input(), any_input()}, consumers_count(1));
    auto unsqueeze_m = wrap_type<ov::op::v0::Unsqueeze>({gather_m, any_input()});

    auto unsqueeze_const_m = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});
    auto concat_m = wrap_type<ov::op::v0::Concat>({unsqueeze_m, unsqueeze_const_m}, consumers_count(1));
    auto concat1_m = wrap_type<ov::op::v0::Concat>({unsqueeze_const_m, unsqueeze_m, any_input()}, consumers_count(1));
    auto bc_m = wrap_type<ov::op::v3::Broadcast>({any_input(), concat_m}, consumers_count(1));
    auto topk_values = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{norm_m, topk_m});
    auto scatter_m = wrap_type<ov::op::v12::ScatterElementsUpdate>({bc_m->output(0), topk_m->output(1), topk_values->output(0), any_input()}, consumers_count(1));
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({scatter_m, any_input()}, consumers_count(1));
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({transpose_m, concat1_m}, consumers_count(1));
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

    // moe compressed
    auto moe_compressed_m = wrap_type<ov::intel_gpu::op::MOECompressed>({hidden_state_m->output(0),
                                                                         unsqueeze_moe_m->output(0),
                                                                         topk_m->output(1),
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
        std::cout << "FuseMOE3GemmCompressed|Begin" << std::endl;
        auto moe_compressed = ov::as_type_ptr<ov::intel_gpu::op::MOECompressed>(pattern_map.at(moe_compressed_m).get_node_shared_ptr());
        if (!moe_compressed || transformation_callback(moe_compressed)) {
            return false;
        }
        OutputVector args(11);
        args[0] = pattern_map.at(hidden_state_m);
        args[1] = pattern_map.at(router_matmul_m);
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
        moe_3gemm_fused_compressed->set_friendly_name(moe_compressed->get_friendly_name());
        ov::copy_runtime_info(moe_compressed, moe_3gemm_fused_compressed);
        ov::replace_node(moe_compressed, moe_3gemm_fused_compressed);
        std::cout << "FuseMOE3GemmCompressed Successfully" << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_compressed_m, "FuseMOE3GemmCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
