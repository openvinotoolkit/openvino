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

struct MoeGatherRef : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::moe_gather::ref")
    explicit MoeGatherRef(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32,
            ov::element::i64,
            ov::element::i8,
            ov::element::u8,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        const auto& input_pshapes = in0_layout.get_partial_shape();

        // Accept rank-2 [tokens, hidden] (Qwen3-style, batch already flattened)
        // and rank-3 [batch, tokens, hidden].  The kernel only needs the last
        // dimension (hidden_size) to be static.
        const auto input_rank = input_pshapes.rank().get_length();
        if ((input_rank != 2 && input_rank != 3) || input_pshapes[input_rank - 1].is_dynamic()) {
            return false;
        }

        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
