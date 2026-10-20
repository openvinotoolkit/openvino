// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "gated_residual_inst.h"
#include "intel_gpu/runtime/layout.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

/// @brief Internal (scratch) buffers used by the systolic (dpas) READ/FINAL_MIX pipeline.
/// Always allocated (mirrors QSAInternalBufIdx) even though only READ/FINAL_MIX nodes
/// that pass `can_use_dpas_gr_read()` ever reference them - see gated_residual.cpp.
struct GatedResidualInternalBufIdx {
    static constexpr size_t RBAR = 0;       ///< f16 [num_rows, C*D], branch-wise RMSNorm output
    static constexpr size_t T_SCRATCH = 1;  ///< f16 [num_rows, L], down-projection + SiLU output

    static constexpr size_t COUNT = 2;
};

/// @brief CM implementation of the gated residual READ/WRITE/FINAL_MIX stages.
/// There is no OpenCL counterpart on purpose: QSA/GR dedicated paths are CM-only.
struct GatedResidualCMImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::gated_residual")
    explicit GatedResidualCMImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : ImplementationManager(cldnn::impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const cldnn::kernel_impl_params& params) const override;

    // READ with `produce_write_gate` has two outputs; without an explicit format for
    // every output the layout selection fails with "No layout format available".
    [[nodiscard]] cldnn::in_out_fmts_t query_formats(const cldnn::program_node& node) const override {
        std::vector<cldnn::format::type> in_fmts(node.get_dependencies().size(), cldnn::format::bfyx);
        std::vector<cldnn::format::type> out_fmts(node.get_outputs_count(), cldnn::format::bfyx);
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override;
};

}  // namespace ov::intel_gpu::cm
