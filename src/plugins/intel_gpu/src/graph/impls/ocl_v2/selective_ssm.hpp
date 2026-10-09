// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "selective_ssm_inst.h"
#include "selective_ssm_jit_utils.hpp"

namespace ov::intel_gpu::ocl {

bool validate_selective_ssm_jit(const cldnn::program_node& node, selective_ssm_jit::device_kind kind);

struct SelectiveSSMOpt : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::selective_ssm::opt")
    explicit SelectiveSSMOpt(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : cldnn::ImplementationManager(cldnn::impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        assert(node.is_type<cldnn::selective_ssm>());
        static constexpr std::array supported_fmts = {cldnn::format::bfyx};
        static constexpr std::array supported_types = {ov::element::f16, ov::element::f32, ov::element::bf16};

        ov::element::Type common_type = ov::element::dynamic;
        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            if (!cldnn::one_of(in_layout.format, supported_fmts) || !cldnn::one_of(in_layout.data_type, supported_types)) {
                return false;
            }
            if (common_type.is_dynamic()) {
                common_type = in_layout.data_type;
            } else if (common_type != in_layout.data_type) {
                return false;
            }
        }

        for (size_t i = 0; i < node.get_outputs_count(); i++) {
            const auto& out_layout = node.get_output_layout(i);
            if (!cldnn::one_of(out_layout.format, supported_fmts) || !cldnn::one_of(out_layout.data_type, supported_types)) {
                return false;
            }
            if (!common_type.is_dynamic() && out_layout.data_type != common_type) {
                return false;
            }
        }
        return true;
    }
};

struct SelectiveSSMJitIntegrated : public SelectiveSSMOpt {
    OV_GPU_PRIMITIVE_IMPL("ocl::selective_ssm::jit_integrated")
    explicit SelectiveSSMJitIntegrated(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr) : SelectiveSSMOpt(shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        return SelectiveSSMOpt::validate_impl(node) && validate_selective_ssm_jit(node, selective_ssm_jit::device_kind::integrated);
    }
};

struct SelectiveSSMJitDiscrete : public SelectiveSSMOpt {
    OV_GPU_PRIMITIVE_IMPL("ocl::selective_ssm::jit_discrete")
    explicit SelectiveSSMJitDiscrete(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr) : SelectiveSSMOpt(shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        return SelectiveSSMOpt::validate_impl(node) && validate_selective_ssm_jit(node, selective_ssm_jit::device_kind::discrete);
    }
};

}  // namespace ov::intel_gpu::ocl
