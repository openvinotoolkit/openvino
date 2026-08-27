// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "paged_selective_ssm_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "selective_ssm_jit_utils.hpp"

namespace ov::intel_gpu::ocl {

bool validate_paged_selective_ssm_jit(const cldnn::program_node& node, selective_ssm_jit::device_kind kind);

struct PagedSelectiveSSMOpt : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_selective_ssm::opt")
    explicit PagedSelectiveSSMOpt(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : cldnn::ImplementationManager(cldnn::impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        assert(node.is_type<cldnn::paged_selective_ssm>());
        static constexpr std::array supported_fmts = {cldnn::format::bfyx};
        static constexpr std::array supported_real_types = {ov::element::f16, ov::element::f32, ov::element::bf16};
        static constexpr std::array supported_index_types = {ov::element::i32, ov::element::i64};

        ov::element::Type data_type = ov::element::dynamic;
        ov::element::Type index_type = ov::element::dynamic;
        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            if (!cldnn::one_of(in_layout.format, supported_fmts))
                return false;
            if (i < cldnn::paged_selective_ssm::RECURRENT_STATE_TABLE) {
                if (!cldnn::one_of(in_layout.data_type, supported_real_types))
                    return false;
                if (data_type.is_dynamic())
                    data_type = in_layout.data_type;
                else if (data_type != in_layout.data_type)
                    return false;
            } else if (i == cldnn::paged_selective_ssm::RECURRENT_STATE_TABLE) {
                if (!cldnn::one_of(in_layout.data_type, supported_real_types))
                    return false;
            } else {
                if (!cldnn::one_of(in_layout.data_type, supported_index_types))
                    return false;
                if (index_type.is_dynamic())
                    index_type = in_layout.data_type;
                else if (index_type != in_layout.data_type)
                    return false;
            }
        }

        const auto& out_layout = node.get_output_layout(0);
        return cldnn::one_of(out_layout.format, supported_fmts) && out_layout.data_type == data_type;
    }
};

struct PagedSelectiveSSMJitIntegrated : public PagedSelectiveSSMOpt {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_selective_ssm::jit_integrated")
    explicit PagedSelectiveSSMJitIntegrated(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : PagedSelectiveSSMOpt(shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        return PagedSelectiveSSMOpt::validate_impl(node) && validate_paged_selective_ssm_jit(node, selective_ssm_jit::device_kind::integrated);
    }
};

struct PagedSelectiveSSMJitDiscrete : public PagedSelectiveSSMOpt {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_selective_ssm::jit_discrete")
    explicit PagedSelectiveSSMJitDiscrete(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr) : PagedSelectiveSSMOpt(shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        return PagedSelectiveSSMOpt::validate_impl(node) && validate_paged_selective_ssm_jit(node, selective_ssm_jit::device_kind::discrete);
    }
};

}  // namespace ov::intel_gpu::ocl
