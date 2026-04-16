// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct GatherMatmulImpl : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::gather_matmul")

    explicit GatherMatmulImpl(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto& device_info = node.get_program().get_engine().get_device_info();
        // Micro-kernel requires systolic (IMMAD) support
        if (!device_info.supports_immad) {
            return false;
        }

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

        auto desc = *(node.get_kernel_impl_params()->typed_desc<gather_matmul>());
        auto layer_id = desc.id;

        size_t input_idx = gather_matmul::BGMInputIdx::INPUT;
        if (!one_of(node.get_input_layout(input_idx).format, supported_fmts) ||
            !one_of(node.get_input_layout(input_idx).data_type, supported_activation_types)) {
            DO_NOT_USE_THIS_KERNEL(layer_id);
        }

        input_idx = gather_matmul::BGMInputIdx::WEIGHT;
        if (!one_of(node.get_input_layout(input_idx).format, supported_fmts) || !one_of(node.get_input_layout(input_idx).data_type, supported_weight_types)) {
            DO_NOT_USE_THIS_KERNEL(layer_id);
        }

        std::vector<cldnn::data_types> quantized_types = {data_types::u4, data_types::i4, data_types::u8, data_types::i8};
        bool has_quant_weight = std::any_of(quantized_types.begin(), quantized_types.end(), [&](const cldnn::data_types& t) -> bool {
            return t == node.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).data_type;
        });

        if (desc.has_bias) {
            input_idx = gather_matmul::BGMInputIdx::BIAS;
            if (!one_of(node.get_input_layout(input_idx).format, supported_fmts) ||
                !one_of(node.get_input_layout(input_idx).data_type, supported_activation_types)) {
                DO_NOT_USE_THIS_KERNEL(layer_id);
            }
        }

        if (has_quant_weight) {
            // GatherMatmulCompressed always has all 6 inputs (A, B, indices, bias_placeholder, scales, zp),
            // even when has_bias=false (bias is a scalar 0 placeholder). Scale is always present at
            // WEIGHT_SCALE. ZP is only present when desc.has_zp; otherwise it is a placeholder to skip.
            std::vector<size_t> quant_param_indices = {gather_matmul::BGMInputIdx::WEIGHT_SCALE};
            if (desc.has_zp)
                quant_param_indices.push_back(gather_matmul::BGMInputIdx::WEIGHT_ZP);
            for (size_t idx : quant_param_indices) {
                const auto& layout = node.get_input_layout(idx);
                if (!one_of(layout.format, supported_fmts) || !one_of(layout.data_type, supported_quant_param_types)) {
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
