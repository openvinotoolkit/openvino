// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/moe.hpp"

#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/moe_gather.hpp>
#include <intel_gpu/primitives/moe_scatter_reduction.hpp>
#include <intel_gpu/primitives/swiglu.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <unordered_map>

#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "intel_gpu/primitives/moe_mask_gen.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/weight_sharing_util.hpp"

namespace ov::intel_gpu {
using namespace cldnn;

// Resolves OTD (offload-to-disk) parameters for a GEMM3_SWIGLU MOECompressed op.
// MOE_OFFLOAD_RATIO specifies the percentage of experts offloaded to disk (0-100).
// The number of GPU-resident LRU slots = num_expert * (100 - ratio) / 100.
// ratio=0 means all resident (no OTD); ratio=100 means all on disk (invalid, disabled).
// Returns true when OTD is enabled (lru_expert_num > 0).
static bool prepare_moe_otd_params(ProgramBuilder& p,
                                   const std::shared_ptr<ov::op::internal::MOECompressed>& op,
                                   std::vector<size_t>& weight_bin_offsets,
                                   std::filesystem::path& weights_path,
                                   size_t& lru_expert_num) {
    using input_idx = cldnn::moe_3gemm_fused_compressed::input_index;
    const auto& config = op->get_config();
    const auto& model = p.get_model();
    const size_t otd_ratio = p.get_config().get_moe_offload_ratio();
    // ratio=0  → all resident, no offload
    // ratio=100 → all on disk, cannot run → treat as disabled
    // otherwise → GPU-resident slots = num_expert * (100 - ratio) / 100
    if (otd_ratio > 0 && otd_ratio < 100) {
        lru_expert_num = std::max<size_t>(1, static_cast<size_t>(config.num_expert) * (100 - otd_ratio) / 100);
    } else {
        lru_expert_num = 0;
    }
    const bool otd_enabled = lru_expert_num > 0;
    if (otd_enabled) {
        weights_path = std::filesystem::path(p.get_config().get_weights_path());
        OPENVINO_ASSERT(!weights_path.empty(),
                        "ov::weights_path property is not set. OTD requires a valid path to the model .bin file. "
                        "Please set ov::weights_path when compiling the model.");
        OPENVINO_ASSERT(std::filesystem::exists(weights_path),
                        "OTD weights file does not exist: ", weights_path);
    }

    auto get_const_offset = [&](size_t index, size_t /*offset_slot*/) -> size_t {
        auto node = op->input_value(index).get_node_shared_ptr();
        auto const_op = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
        OPENVINO_ASSERT(const_op != nullptr, "Expected constant input for MOE3GemmFusedCompressed");
        const auto& rt_info = const_op->get_rt_info();
        auto attr_it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());

        if (attr_it != rt_info.end()) {
            return attr_it->second.as<ov::WeightlessCacheAttribute>().bin_offset;
        }

        // Buffer descriptor offset (works when constant data is mmap'd from bin file).
        auto source_buf = ov::weight_sharing::Extension::get_constant_source_buffer(*const_op);
        OPENVINO_ASSERT(source_buf,
                        "OTD: Cannot determine bin offset for MOE weight constant. "
                        "Neither WeightlessCacheAttribute nor mmap source buffer is available.");
        return ov::weight_sharing::Extension::get_constant_id(*const_op);
    };

    const std::array<size_t, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> const_input_idx_by_offset = {
        static_cast<size_t>(input_idx::weight_0),
        static_cast<size_t>(input_idx::weight_1),
        static_cast<size_t>(input_idx::weight_2),
        static_cast<size_t>(input_idx::scale_0),
        static_cast<size_t>(input_idx::scale_1),
        static_cast<size_t>(input_idx::scale_2),
        static_cast<size_t>(input_idx::zp_0),
        static_cast<size_t>(input_idx::zp_1),
        static_cast<size_t>(input_idx::zp_2)
    };

    weight_bin_offsets.assign(cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count, 0);
    // Serialized offsets are only needed for OTD path (weight-on-demand loading).
    if (otd_enabled) {
        for (size_t i = 0; i < const_input_idx_by_offset.size(); i++) {
            weight_bin_offsets[i] = get_const_offset(const_input_idx_by_offset[i], i);
        }
    }
    return otd_enabled;
}

static void CreateMOECompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOECompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    auto& config = op->get_config();
    std::vector<cldnn::input_info> input_infos;
    for (const auto& input : inputs) {
        input_infos.push_back(cldnn::input_info(input));
    }
    if (config.expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
        const size_t base_inputs = 12;
        const size_t shared_inputs = config.num_shared_expert > 0 ? 10 : 0;
        const size_t expected_inputs = base_inputs + shared_inputs;
        validate_inputs_count(op, {expected_inputs});

        // Resolve OTD (offload-to-disk) parameters; no-op when MOE_OFFLOAD_RATIO == 0.
        std::vector<size_t> weight_bin_offsets;
        std::filesystem::path weights_path;
        size_t lru_expert_num = 0;
        prepare_moe_otd_params(p, op, weight_bin_offsets, weights_path, lru_expert_num);

        const std::string layerName = layer_type_name_ID(op);
        const cldnn::moe_3gemm_fused_compressed moe(layerName, input_infos, config, weight_bin_offsets, weights_path, lru_expert_num);
        p.add_primitive(*op, moe);
    } else {
        // Create GEMM2_BIAS_SWIGLU_CLAMP specific primitives
        // input0 : input {#tokens, hidden_size}
        // input1 : topk_weight {#tokens, num_experts_per_token}
        // input2 : topk_idx {#tokens, num_experts_per_token}
        // input3 : compressed_weights_input_up {#experts, ofm, num_groups, group_size}
        // input4 : scale_input_up {#experts, ofm, num_groups, 1}
        // input5 : bias_up {#experts, 1, ofm}
        // input6 : compressed_weights_input_down {#experts, ofm, num_groups, group_size}
        // input7 : scale_input_down {#experts, ofm, num_groups, 1}
        // input8 : bias_down {#experts, 1, ofm}
        // moe_mask_gen
        // moe_gather
        // moe_gemm_up + bias
        // swiglu_with_clamp
        // moe_gemm_down + bias
        // moe_scatter_reduce
        std::string prim_name_base = layer_type_name_ID(op);
        auto moe_mask_gen_name = prim_name_base + "_moe_mask_gen";
        auto moe_mask_gen_reshape_name = prim_name_base + "_moe_mask_gen_reshape";
        auto moe_gather_name = prim_name_base + "_moe_gather";
        auto moe_bias_up_name = prim_name_base + "_moe_bias_up";
        auto moe_gemm_up_name = prim_name_base + "_moe_gemm_up";
        auto moe_swiglu_name = prim_name_base + "_moe_swiglu";
        auto moe_gemm_down_name = prim_name_base + "_moe_gemm_down";
        auto moe_bias_down_name = prim_name_base + "_moe_bias_down";
        auto moe_scatter_reduce_name = prim_name_base + "_moe_scatter_reduce";
        auto moe_mask_gen_prim = cldnn::moe_mask_gen(moe_mask_gen_name,
                                                     input_infos[2],  // topk indices
                                                     static_cast<int32_t>(config.num_expert),
                                                     static_cast<int32_t>(config.top_k),
                                                     true);
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
            input_infos[3],               // compressed_weights_input_up
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT)};
        size_t down_idx = 0;
        if (config.has_zp) {
            moe_gemm_up_inputs.push_back(input_infos[6]);  // bias_up
            moe_gemm_up_inputs.push_back(input_infos[4]);  // scale_input_up
            moe_gemm_up_inputs.push_back(input_infos[5]);  // zp_input_up
            down_idx = 7;
        } else {
            moe_gemm_up_inputs.push_back(input_infos[5]);  // bias_up
            moe_gemm_up_inputs.push_back(input_infos[4]);  // scale_input_up
            down_idx = 6;
        }
        auto moe_gemm_up = cldnn::moe_gemm(moe_gemm_up_name, moe_gemm_up_inputs, config);
        moe_gemm_up.has_bias = true;
        p.add_primitive(*op, moe_gemm_up);

        // GPT-OSS swiglu: stride-2 interleave (gate=swish, up=clamp+add).
        auto moe_swiglu_prim = cldnn::swiglu(moe_swiglu_name,
                                             input_info(moe_gemm_up_name),
                                             2,  // axis
                                             2,  // glu_stride
                                             ov::op::internal::GLU::GluType::Swish,
                                             config.gate_idx,
                                             -config.expert_alpha,  // clamp_min
                                             config.expert_alpha,   // clamp_max
                                             config.expert_beta,    // swish beta
                                             1.0f,                  // up_add_val
                                             cldnn::tensor(),
                                             config.scale_factor.value_or(-1.0f));   // activations scaling
        p.add_primitive(*op, moe_swiglu_prim);
        std::vector<cldnn::input_info> moe_gemm_down_inputs = {
            input_info(moe_swiglu_name),
            input_infos[down_idx],  // compressed_weights_input_down
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
        };

        if (config.has_zp) {
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 3]);  // bias_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 1]);  // scale_input_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 2]);  // zp_input_up
        } else {
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 2]);  // bias_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 1]);  // scale_input_up
        }

        auto moe_gemm_down = cldnn::moe_gemm(moe_gemm_down_name, moe_gemm_down_inputs, config);
        moe_gemm_down.has_bias = true;
        p.add_primitive(*op, moe_gemm_down);
        auto moe_scatter_reduce_prim =
            cldnn::moe_scatter_reduction(moe_scatter_reduce_name,
                                         input_info(moe_gemm_down_name),
                                         input_infos[2],
                                         input_infos[1],
                                         input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_PER_EXPERT),
                                         input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
                                         input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
                                         input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
                                         config,
                                         true);
        p.add_primitive(*op, moe_scatter_reduce_prim);
    }
}

REGISTER_FACTORY_IMPL(internal, MOECompressed);

}  // namespace ov::intel_gpu
