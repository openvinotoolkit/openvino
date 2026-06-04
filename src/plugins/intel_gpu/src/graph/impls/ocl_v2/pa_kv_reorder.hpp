// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;

namespace ov::intel_gpu::ocl {

struct PA_KV_reorder : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::pa_kv_reorder")
    explicit PA_KV_reorder(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        if (node.has_fused_primitives())
            return false;
        // OCL kernel expects head-major K layout [num_blocks, num_kv_heads, head_size, BLOCK_SIZE].
        // The CM PA pipeline materializes a token-major layout instead — for that, defer to
        // the CM reorder impl.
        const auto& key_layout = node.get_input_layout(cldnn::pa_kv_reorder::PaKVReorderInputIdx::KEY_CACHE);
        const auto& key_shape = key_layout.get_partial_shape();
        if (key_shape.size() == 4 && key_shape[2].is_static() && key_shape[3].is_static()) {
            const size_t shape2 = static_cast<size_t>(key_shape[2].get_length());
            const size_t shape3 = static_cast<size_t>(key_shape[3].get_length());
            // Plain PA: shape[2] = head_size, shape[3] = block_size = 16. Token-major would
            // have shape[2] >= 128. Reject token-major here so CM impl picks it up.
            if (shape2 > shape3) {
                return false;
            }
        }
        return false; // force to cm kernel path temporarily
    }
};

}  // namespace ov::intel_gpu::ocl
