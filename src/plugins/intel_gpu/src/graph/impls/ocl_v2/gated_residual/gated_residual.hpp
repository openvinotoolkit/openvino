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

namespace ov::intel_gpu::ocl {

/// @brief Internal (scratch) buffers of the 3-dispatch systolic READ/FINAL_MIX pipeline.
struct GatedResidualInternalBufIdx {
    static constexpr size_t RBAR = 0;       ///< f16 [num_rows, C*D], branch-wise RMSNorm output
    static constexpr size_t T_SCRATCH = 1;  ///< f16 [num_rows, L], down-projection + SiLU output
};

/// @brief OpenCL implementation of the gated residual READ/WRITE/FINAL_MIX stages.
struct GatedResidualImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::gated_residual")
    explicit GatedResidualImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : ImplementationManager(cldnn::impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const cldnn::kernel_impl_params& params) const override;

    // READ with `produce_write_gate` has two outputs; every output needs an explicit format.
    [[nodiscard]] cldnn::in_out_fmts_t query_formats(const cldnn::program_node& node) const override {
        std::vector<cldnn::format::type> in_fmts(node.get_dependencies().size(), cldnn::format::bfyx);
        std::vector<cldnn::format::type> out_fmts(node.get_outputs_count(), cldnn::format::bfyx);
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override;
};

}  // namespace ov::intel_gpu::ocl
