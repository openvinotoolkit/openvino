// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "fully_connected_inst.h"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "openvino/core/type/element_type.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// Single ImplementationManager covering the whole GGUF weight-only-quantised FullyConnected family.
//
// A GGUF weight enters the graph as an op::v0::Constant whose element type is one of the opaque
// element::gguf_* block types (raw llama.cpp block bytes, shape [N, K]); scale and zero-point are
// embedded inside each block, so the FullyConnectedCompressed weight_scales / weight_zero_points
// inputs are left empty. This manager validates such nodes and produces an OCL kernel that decodes
// the blocks in registers and computes X * W^T directly (see fc_gguf_opt.cl). The packed format is
// selected per node from weights.data_type via a JIT constant, so one manager / one registry entry
// serves every GGUF format (cf. SPEC.md §4.1, SUMMARY.md §3.1).
//
// PR-GPU baseline white-list: only the five formats that cover qwen3's common quantisation recipes
// (Q4_K_M / Q5_K_M / Q6_K / Q8_0) have a kernel. The remaining 18 element types are intentionally
// rejected by validate_impl with a clear "not yet implemented" signal (SPEC.md §4.1 / D7); the
// element types themselves are fully defined in OV Core so the ABI surface stays stable.
struct FCGGUFOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::fc_gguf_opt")

    explicit FCGGUFOpt(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    // GGUF formats with a working native kernel in this release (SPEC.md §4.1).
    static constexpr std::array<element::Type_t, 5> kSupportedBaseline = {
        element::Type_t::gguf_q4_0,
        element::Type_t::gguf_q4_k,
        element::Type_t::gguf_q5_k,
        element::Type_t::gguf_q6_k,
        element::Type_t::gguf_q8_0,
    };

    static bool is_supported_baseline(element::Type_t t) {
        for (auto s : kSupportedBaseline) {
            if (s == t) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                              const RuntimeParams& params) const override;

    // Program-level check (run once at graph compilation). Accepts a fully_connected node whose
    // weight (input 1) is a baseline GGUF block type. Does NOT inspect the scale/ZP inputs: for GGUF
    // they are intentionally empty (scale lives inside the block), unlike the W4A16/W4A8 path.
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());

        const auto& desc = *node.get_kernel_impl_params()->typed_desc<fully_connected>();
        if (!desc.compressed_weights) {
            return false;
        }

        const auto& in0 = node.get_input_layout(0);  // activation
        const auto& in1 = node.get_input_layout(1);  // weight

        if (in0.data_type != data_types::f16 && in0.data_type != data_types::f32) {
            return false;
        }
        if (!element::is_gguf_block(in1.data_type)) {
            return false;
        }
        // Element type is defined in OV Core but the kernel for it ships in a later increment.
        return is_supported_baseline(in1.data_type);
    }

    // Shape check used both for the shape-agnostic (dynamic) binding and the per-concrete-shape
    // runtime specialisation. The weight is always a static GGUF Constant; the activation may be
    // dynamic (compile time) or concrete (runtime). The kernel handles any M (decode GEMV and prefill
    // GEMM), so accept the dynamic-activation case and only validate K alignment once concrete.
    [[nodiscard]] bool support_shapes(const kernel_impl_params& params) const override {
        const auto& in1 = params.get_input_layout(1);
        if (in1.is_dynamic() || !element::is_gguf_block(in1.data_type)) {
            return false;
        }
        const auto& in0 = params.get_input_layout(0);
        if (in0.is_dynamic()) {
            return true;
        }

        const auto& shape_a = in0.get_shape();
        if (shape_a.size() < 2) {
            return false;
        }
        const size_t K = shape_a[shape_a.size() - 1];
        const size_t block_elem = element::Type(in1.data_type).block_elem_count();
        return block_elem != 0 && K % block_elem == 0;
    }
};

}  // namespace ov::intel_gpu::ocl
