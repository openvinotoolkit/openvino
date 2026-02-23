// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct MoEGemm : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::moe_gemm")

    explicit MoEGemm(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto& device_info = node.get_program().get_engine().get_device_info();
        // TODO: only micro kernel is available now for moe_gemm
        if (!device_info.supports_immad)
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
};

}  // namespace ov::intel_gpu::ocl
