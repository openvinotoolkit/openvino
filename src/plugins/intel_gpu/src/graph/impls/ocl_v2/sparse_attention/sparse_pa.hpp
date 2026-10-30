// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "sparse_pa_inst.h"

namespace ov::intel_gpu::ocl {

/// @brief OpenCL implementation of the SparsePA primitive: Q0 (K/V cache write) + Q3 (selected-block attention with the
/// automatically derived causal tail). Q0 writes directly into the caller-owned KEY_CACHE/VALUE_CACHE inputs and Q3
/// reads the paired indexer's selection inputs directly.
struct SparsePAImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::sparse_pa")
    explicit SparsePAImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : ImplementationManager(cldnn::impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const cldnn::kernel_impl_params& params) const override;

    [[nodiscard]] cldnn::in_out_fmts_t query_formats(const cldnn::program_node& node) const override {
        std::vector<cldnn::format::type> in_fmts(node.get_dependencies().size(), cldnn::format::bfyx);
        std::vector<cldnn::format::type> out_fmts(node.get_outputs_count(), cldnn::format::bfyx);
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override;
};

}  // namespace ov::intel_gpu::ocl
