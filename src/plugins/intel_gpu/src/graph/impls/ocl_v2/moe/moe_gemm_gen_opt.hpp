// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "moe_gemm_base.hpp"
#include "moe_gemm_inst.h"
#include "ocl_v2/utils/jitter.hpp"
using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

using RuntimeParams = kernel_impl_params;

struct moe_config {
    bool has_bias = false;
    bool is_activation_quantized = false;
    bool is_activation_symmetric_quantized = false;
    bool is_weight_quantized = false;
    bool is_weight_symmetric_quantized = false;
    int32_t weight_group_size = -1;
    int32_t weight_scale_idx = -1;
    int32_t weight_zp_idx = -1;
    bool has_batch_dim = false;
};

class MoEGemmOptGeneratorBase : public MoEGemmBase {
public:
    MoEGemmOptGeneratorBase(std::string_view name, std::string_view stage) : MoEGemmBase(name, stage) {}

    static moe_config get_moe_cfg(const kernel_impl_params& params) {
        moe_config moe_cfg;
        auto desc = params.typed_desc<moe_gemm>();
        std::vector<cldnn::data_types> quantized_types = {data_types::u4, data_types::i4, data_types::u8, data_types::i8};
        moe_cfg.has_bias = desc->has_bias;
        if (std::any_of(quantized_types.begin(), quantized_types.end(), [=](const cldnn::data_types& t) -> bool {
                return t == params.input_layouts[1].data_type;
            })) {
            moe_cfg.is_weight_quantized = true;
            if (desc->has_bias) {
                moe_cfg.weight_scale_idx = moe_gemm::MoEGemmInputIdx::WEIGHT_SCALE;
                moe_cfg.weight_zp_idx = moe_gemm::MoEGemmInputIdx::WEIGHT_ZP;
            } else {
                moe_cfg.weight_scale_idx = moe_gemm::MoEGemmInputIdx::WEIGHT_SCALE - 1;
                moe_cfg.weight_zp_idx = moe_gemm::MoEGemmInputIdx::WEIGHT_ZP - 1;
            }
            const auto& weight_shape = params.input_layouts[moe_gemm::MoEGemmInputIdx::WEIGHT].get_shape();
            // experts weight : [#experts, ofm, num_groups, group_size]
            auto k = (weight_shape.size() == 4) ? weight_shape[2] * weight_shape[3] : weight_shape[2];
            auto scale_group_dim = params.input_layouts[moe_cfg.weight_scale_idx].get_shape().size() - 2;
            auto num_scale_groups = (weight_shape.size() == 4) ? params.input_layouts[moe_cfg.weight_scale_idx].get_shape()[scale_group_dim] : 1;
            moe_cfg.weight_group_size = k / num_scale_groups;
            if (static_cast<int32_t>(params.input_layouts.size()) > moe_cfg.weight_zp_idx) {
                moe_cfg.is_weight_symmetric_quantized = false;
            } else {
                moe_cfg.is_weight_symmetric_quantized = true;
            }
        }
        moe_cfg.has_batch_dim = desc->has_batch_dim;
        return moe_cfg;
    }
};
}  // namespace ov::intel_gpu::ocl
