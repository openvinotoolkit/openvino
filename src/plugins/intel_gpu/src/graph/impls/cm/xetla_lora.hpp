// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <utility>

#include "common_utils/jitter.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "lora_inst.h"
#include "registry/implementation_manager.hpp"
#include "xetla_postops.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

struct LoRAImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::lora")
    explicit LoRAImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<lora>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::bfyx);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::bfyx);
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<lora>());

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        const auto& info = engine.get_device_info();

        if (!check_cm_jit_support(engine, config) || info.arch != gpu_arch::xe2 || !config.get_use_cm()) {
            return false;
        }

        const auto& out_layout = node.get_output_layout(0);
        const auto& in0_layout = node.get_input_layout(0);

        static constexpr std::array supported_fmts = {format::bfyx};
        static constexpr std::array supported_types = {ov::element::f16, ov::element::bf16};

        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        const auto lora_count = ((node.get_inputs_count() - 2ul) / 3ul);
        if (lora_count != 1) {
            return false;
        }

        for (auto& prim : node.get_fused_primitives()) {
            const bool is_eltwise = fused_ops_are_one_of<eltwise>({prim});
            const bool is_activation = fused_ops_are_one_of<activation>({prim});

            if (is_activation) {
                const auto activation_desc = std::static_pointer_cast<const activation>(prim.desc);
                const auto xetla_activation_func = get_xetla_activation_op(activation_desc->activation_function);

                if (Activation::ActivationOp::none == xetla_activation_func) {
                    return false;
                }
                if (!((activation_desc->additional_params.a == 1.0f) && (activation_desc->additional_params.b == 0.0f))) {
                    return false;
                }
                if (prim.deps.size() != 0) {
                    return false;
                }

            } else if (is_eltwise) {
                const auto eltwise_desc = std::static_pointer_cast<const eltwise>(prim.desc);
                const auto& eltwise_in_layout = prim.input_layout;
                const auto xetla_eltwise_mode = get_xetla_eltwise_op(eltwise_desc->mode);
                if (Eltwise::EltwiseOp::none == xetla_eltwise_mode) {
                    return false;
                }

                if(eltwise_desc->broadcast_spec.m_axis != 0) {
                    return false;
                }

            } else {
                return false;
            }
        }
        return true;
    }
};

}  // namespace ov::intel_gpu::cm
