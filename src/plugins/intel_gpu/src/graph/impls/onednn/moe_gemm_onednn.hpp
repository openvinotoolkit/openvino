// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifndef ENABLE_ONEDNN_FOR_GPU
    #define ENABLE_ONEDNN_FOR_GPU 1
#endif
#include "moe_gemm_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

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

struct MoEGemmImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::moe_gemm")
    MoEGemmImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<moe_gemm>());
        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown || !config.get_use_onednn())
            return false;

        static const std::vector<format> supported_fmts = {
            format::bfyx,
        };

        static const std::vector<ov::element::Type_t> supported_activation_types = {
            ov::element::f16,
            ov::element::i8,
            ov::element::u8,
        };

        static const std::vector<ov::element::Type_t> supported_weight_types = {
            ov::element::f16,
            ov::element::u4,
            ov::element::i4,
            ov::element::i8,
            ov::element::u8,
        };

        static const std::vector<ov::element::Type_t> supported_quant_param_types = {
            ov::element::f16,
            ov::element::u4,
            ov::element::i4,
            ov::element::i8,
            ov::element::u8,
            ov::element::i32,
        };

        static const std::vector<ov::element::Type_t> supported_mask_types = {
            ov::element::i32,
        };

        auto desc = *(node.get_kernel_impl_params()->typed_desc<moe_gemm>());
        auto layer_id = desc.id;

        size_t input_idx = moe_gemm::MoEGemmInputIdx::INPUT;
        if (!one_of(node.get_input_layout(input_idx).format, supported_fmts) ||
            !one_of(node.get_input_layout(input_idx).data_type, supported_activation_types)) {
            DO_NOT_USE_THIS_KERNEL(layer_id);
        }

        input_idx = moe_gemm::MoEGemmInputIdx::WEIGHT;
        if (!one_of(node.get_input_layout(input_idx).format, supported_fmts) || !one_of(node.get_input_layout(input_idx).data_type, supported_weight_types)) {
            DO_NOT_USE_THIS_KERNEL(layer_id);
        }

        std::vector<cldnn::data_types> quantized_types = {data_types::u4, data_types::i4, data_types::u8, data_types::i8};
        bool has_quant_weight = false;
        if (std::any_of(quantized_types.begin(), quantized_types.end(), [&](const cldnn::data_types& t) -> bool {
                return t == node.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT).data_type;
            })) {
            has_quant_weight = true;
        }

        input_idx = moe_gemm::MoEGemmInputIdx::BIAS;
        if (desc.has_bias) {
            if (!one_of(node.get_input_layout(input_idx).format, supported_fmts) ||
                !one_of(node.get_input_layout(input_idx).data_type, supported_activation_types)) {
                DO_NOT_USE_THIS_KERNEL(layer_id);
            }
        }

        if (has_quant_weight) {
            size_t quant_params_idx_start =
                desc.has_bias ? static_cast<size_t>(moe_gemm::MoEGemmInputIdx::WEIGHT_SCALE) : static_cast<size_t>(moe_gemm::MoEGemmInputIdx::WEIGHT_SCALE - 1);
            for (size_t i = quant_params_idx_start; i < node.get_input_layouts().size(); i++) {
                if (!one_of(node.get_input_layout(i).format, supported_fmts) || !one_of(node.get_input_layout(i).data_type, supported_quant_param_types)) {
                    DO_NOT_USE_THIS_KERNEL(layer_id);
                }
            }
        }

        const auto& output_layout = node.get_output_layout(0);
        if (!one_of(output_layout.format, supported_fmts) || !one_of(output_layout.data_type, supported_activation_types)) {
            DO_NOT_USE_THIS_KERNEL(layer_id);
        }

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<moe_gemm>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            size_t out_rank = node.get_output_layout().get_rank();
            auto target_format = format::get_default_format(out_rank);

            in_fmts[idx] = target_format;

            if (out_fmts[0] == format::any) {
                out_fmts[0] = target_format;
            }
        }

        return {in_fmts, out_fmts};
    }

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

}  // namespace onednn
}  // namespace cldnn
