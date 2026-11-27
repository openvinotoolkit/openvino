// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_gpu/primitives/reorder.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

namespace cldnn {
namespace ocl {

// Packs the ternary u2 weights of a TernOCL FullyConnected into format::is_os_yx_isv16_ternary.
// The FC impl requests this layout through its weights reorder params, so post_optimize_weights
// inserts the reorder and propagate_constants folds it at build time: the packed weights are an
// ordinary constant of the program and the source constant is left untouched for any other user.
// The packing runs on the host (it also checks that every code is ternary), hence impl_types::cpu.
struct TernoclInt2WeightsReorderImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("TernoclInt2WeightsReorderImplementationManager")
    explicit TernoclInt2WeightsReorderImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cpu, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    std::unique_ptr<primitive_impl> create_impl(const kernel_impl_params& params) const override;
    in_out_fmts_t query_formats(const program_node& node) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<reorder>());
        const auto& in = node.get_input_layout(0);
        const auto& out = node.get_output_layout(0);
        return out.format == format::is_os_yx_isv16_ternary && out.data_type == data_types::u2 && in.data_type == data_types::u2 &&
               (in.format == format::bfyx || in.format == format::oiyx) && !in.data_padding && in.is_static();
    }
};

}  // namespace ocl
}  // namespace cldnn
