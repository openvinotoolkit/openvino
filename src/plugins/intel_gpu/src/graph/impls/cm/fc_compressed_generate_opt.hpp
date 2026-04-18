// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "fully_connected_inst.h"
#include "impls/ocl/kernel_selector_helper.h"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

// CM GEMV kernel for W4A16/W4A8 weight-only-quantised fully-connected.
// Same algorithmic design as ocl::FCCompressedGenerateOpt but compiled via
// the CM compiler (-cmc) for potentially tighter ISA on Xe2+ architectures.
struct FCCompressedGenerateOptCM : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::fc_compressed_generate_opt")

    // Register as impl_types::ocl so the multi-impl pool's runtime switching
    // heuristic (which only understands onednn/ocl) can select this kernel.
    // The CM compiler is invoked via KernelLanguage::CM + "-cmc" build options,
    // independent of the impl_type enum.
    explicit FCCompressedGenerateOptCM(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                              const kernel_impl_params& params) const override;

    bool raw_sub_byte_weight_compatible() const noexcept override { return true; }

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        // Temporarily disabled: OCL SLM-cooperative GEMV kernel is being actively optimised.
        // CM registers as impl_types::ocl and steals the OCL slot in the multi-impl pool
        // (seen-set dedup in primitive_inst.cpp ~line 3614), preventing OCL from being selected.
        return false;

        assert(node.is_type<fully_connected>());

        // CM requires CM JIT support, IMMAD (systolic), and use_cm flag.
        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        const auto& info = engine.get_device_info();
        if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
            return false;
        }

        const auto& desc = *node.get_kernel_impl_params()->typed_desc<fully_connected>();

        if (!desc.compressed_weights)
            return false;
        if (!desc.decompression_scale.is_valid())
            return false;

        const auto& in0 = node.get_input_layout(0);   // activation
        const auto& in1 = node.get_input_layout(1);   // weight

        const bool act_is_f16 = (in0.data_type == data_types::f16);
        const bool act_is_i8  = (in0.data_type == data_types::i8);
        if (!act_is_f16 && !act_is_i8)
            return false;

        if (in1.data_type != data_types::u4 && in1.data_type != data_types::i4)
            return false;

        const bool has_bias = desc.bias.is_valid();
        const size_t scale_idx = has_bias ? 3 : 2;
        if (scale_idx >= node.get_input_layouts().size())
            return false;
        if (node.get_input_layout(scale_idx).data_type != data_types::f16)
            return false;

        if (act_is_f16) {
            if (desc.dynamic_quantized_activation)
                return false;
        } else {
            if (!desc.dynamic_quantized_activation)
                return false;
            if (!desc.activation_scale.is_valid())
                return false;
            if (desc.activation_zero_point.is_valid())
                return false;
            const bool has_weight_zp = desc.decompression_zero_point.is_valid();
            const size_t act_scale_idx = scale_idx + 1 + (has_weight_zp ? 1 : 0);
            if (act_scale_idx >= node.get_input_layouts().size())
                return false;
            if (node.get_input_layout(act_scale_idx).data_type != data_types::f16)
                return false;
        }

        return true;
    }

    [[nodiscard]] bool support_shapes(const kernel_impl_params& params) const override {
        const auto& in0 = params.get_input_layout(0);
        if (in0.is_dynamic())
            return false;

        const auto& shape = in0.get_shape();
        const size_t rank = shape.size();
        if (rank < 2)
            return false;

        const size_t M = shape[rank - 2];
        if (M != 1)
            return false;

        const size_t K = shape[rank - 1];
        if (params.input_layouts.size() < 3)
            return false;
        const auto& scale_layout = params.input_layouts[2];
        if (scale_layout.is_dynamic())
            return false;
        const auto& scale_shape = scale_layout.get_shape();
        if (scale_shape.size() < 2)
            return false;
        const size_t num_groups = scale_shape[scale_shape.size() - 1];
        if (num_groups == 0 || K % num_groups != 0)
            return false;
        const size_t group_size = K / num_groups;
        constexpr size_t VEC_SIZE_CHECK = 8;
        if (group_size % VEC_SIZE_CHECK != 0)
            return false;
        if (K % group_size != 0)
            return false;

        return true;
    }
};

}  // namespace ov::intel_gpu::cm
