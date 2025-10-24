// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/constant.hpp"
#include "openvino/op/moe.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

#include "intel_gpu/primitives/moe_gemm.hpp"
#include "intel_gpu/primitives/moe_mask_gen.hpp"
#include <intel_gpu/primitives/moe_scatter_reduction.hpp>
#include <intel_gpu/primitives/moe_gather.hpp>
#include <intel_gpu/primitives/swiglu.hpp>
#include <intel_gpu/primitives/eltwise.hpp>

#include <limits>

namespace ov {
namespace op {
namespace internal {
using MOECompressed = ov::intel_gpu::op::MOECompressed;
}  // namespace internal
}  // namespace op
}  // namespace ov


namespace ov::intel_gpu {
using namespace cldnn;
static void CreateMOECompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOECompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    auto& config = op->get_config();
    std::vector<cldnn::input_info> input_infos;
    for (const auto& input : inputs) {
        input_infos.push_back(cldnn::input_info(input));
    }
    if (config.expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
        // Create GEMM3_SWIGLU specific primitives
    } else  {
        // Create GEMM2_BIAS_SWIGLU_CLAMP specific primitives
        std::cout << "create moe op!" << std::endl;
        // input0 : input {#tokens, hidden_size}
        // input1 : topk_weight {#tokens, num_experts_per_token}
        // input2 : topk_idx {#tokens, num_experts_per_token}
        // input3 : compressed_weights_input_up {#experts, ofm, num_groups, group_size}
        // input4 : scale_input_up {#experts, ofm, num_groups, 1}
        // input5 : bias_up {#experts, 1, ofm}
        // input6 : compressed_weights_input_down {#experts, ofm, num_groups, group_size}
        // input7 : scale_input_down {#experts, ofm, num_groups, 1}
        // input8 : bias_down {#experts, 1, ofm}
        // mask
        // gather
        // gemm_up
        // bias
        // swiglu
        // gemm_down
        // bias
        // scatter_reduce
        std::string prim_name_base = layer_type_name_ID(op);
        auto  moe_mask_gen_name = prim_name_base + "_moe_mask_gen";
        auto  moe_mask_gen_reshape_name = prim_name_base + "_moe_mask_gen_reshape";
        auto  moe_gather_name = prim_name_base + "_moe_gather";
        auto  moe_bias_up_name = prim_name_base + "_moe_bias_up";
        auto  moe_gemm_up_name = prim_name_base + "_moe_gemm_up";
        auto  moe_swiglu_name = prim_name_base + "_moe_swiglu";
        auto  moe_gemm_down_name = prim_name_base + "_moe_gemm_down";
        auto  moe_bias_down_name = prim_name_base + "_moe_bias_down";
        auto  moe_scatter_reduce_name = prim_name_base + "_moe_scatter_reduce";
        auto moe_mask_gen_prim = cldnn::moe_mask_gen(moe_mask_gen_name,
                                                     input_infos[2],  // topk indices
                                                     static_cast<uint32_t>(config.num_expert),
                                                     static_cast<uint32_t>(config.top_k));
        p.add_primitive(*op, moe_mask_gen_prim);
        auto moe_mask_gen_reshape_prim =
            cldnn::moe_mask_gen_reshape(moe_mask_gen_reshape_name,
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::TOKENS_PER_EXPERT),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::EXPERTS_INFO_START_IDX),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::EXPERTS_ID),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::TOKENS_LENS_PER_EXPERT),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::NUM_ACTUALLY_USED_EXPERTS));
        p.add_primitive(*op, moe_mask_gen_reshape_prim);
        auto moe_gather_prim = cldnn::moe_gather(moe_gather_name,
                                                 input_infos[0],  // input
                                                 input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_PER_EXPERT),
                                                 config);
        p.add_primitive(*op, moe_gather_prim);
        std::vector<cldnn::input_info> moe_gemm_up_inputs = {
            input_info(moe_gather_name),  // topk_weight
            input_infos[3],  // compressed_weights_input_up
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
            input_infos[5],  // bias_up
            input_infos[4],  // scale_input_up
        };
        auto moe_gemm_up = cldnn::moe_gemm(moe_gemm_up_name, moe_gemm_up_inputs, config.top_k);
        moe_gemm_up.has_bias = true;
        p.add_primitive(*op, moe_gemm_up);
        // config.expert_alpha : clamp_max
        // config.expert_beta : swish_beta
        auto moe_swiglu_prim = cldnn::swiglu(moe_swiglu_name,
                                             input_info(moe_gemm_up_name),
                                             2,  // axis
                                             config.hidden_size,
                                             ov::op::internal::GLU::GluType::Swish,
                                             1,                    // gate : second half
                                             std::numeric_limits<float>::min(),   // clamp_min
                                             config.expert_alpha,  // clamp_max
                                             config.expert_beta,   // swish beta
                                             cldnn::tensor());
        p.add_primitive(*op, moe_swiglu_prim);
        std::vector<cldnn::input_info> moe_gemm_down_inputs = {
            input_info(moe_swiglu_name),
            input_infos[6],  // compressed_weights_input_down
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
            input_infos[8],  // bias_down
            input_infos[7],  // scale_input_down
        };
        auto moe_gemm_down = cldnn::moe_gemm(moe_gemm_down_name, moe_gemm_down_inputs, config.top_k);
        moe_gemm_down.has_bias = true;
        p.add_primitive(*op, moe_gemm_down);
        auto moe_scatter_reduce_prim = cldnn::moe_scatter_reduction(moe_scatter_reduce_name,
                input_info(moe_gemm_down_name),
                input_infos[2],
                input_infos[1],
                input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_PER_EXPERT),
                input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
                input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
                config);
        p.add_primitive(*op, moe_scatter_reduce_prim);
    }
}

REGISTER_FACTORY_IMPL(internal, MOECompressed);

}  // namespace ov::intel_gpu
