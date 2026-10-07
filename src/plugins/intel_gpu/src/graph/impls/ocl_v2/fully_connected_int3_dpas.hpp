// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// FullyConnected with u3 compressed weights and int8 dynamically quantized activations on the matrix engine.
// An impl holds the activation quantizer plus the GEMM variants tuned for row counts: a static impl the one for its
// rows, a dynamic impl all of them, so that no kernel is compiled at runtime.
struct FullyConnectedInt3Dpas : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::fc_int3_dpas")
    explicit FullyConnectedInt3Dpas(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override;
    [[nodiscard]] bool support_shapes(const RuntimeParams& params) const override;
    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override;
    [[nodiscard]] bool validate_when_forced() const override {
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
