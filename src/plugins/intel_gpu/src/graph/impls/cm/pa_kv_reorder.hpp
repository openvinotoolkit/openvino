// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

struct PaKVReorderImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::pa_kv_reorder")
    explicit PaKVReorderImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : ImplementationManager(cldnn::impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const cldnn::kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        if (node.has_fused_primitives()) {
            return false;
        }
        const auto desc = node.as<cldnn::pa_kv_reorder>().get_primitive();
        // CM reorder kernel handles only fp16 / int8 caches (matches CM PA validate_impl).
        if (!(desc->cache_dt == cldnn::data_types::f16 || desc->cache_dt == cldnn::data_types::i8 || desc->cache_dt == cldnn::data_types::u8)) {
            return false;
        }

        // CM PA writes K cache token-major: [num_blocks, num_kv_heads, BLOCK_SIZE_with_tail, head_size].
        // OCL plain PA writes head-major:   [num_blocks, num_kv_heads, head_size, BLOCK_SIZE].
        // Distinguish by checking which inner dim is BLOCK_SIZE-aligned to the CM xattn block size (256).
        const auto& key_layout = node.get_input_layout(cldnn::pa_kv_reorder::PaKVReorderInputIdx::KEY_CACHE);
        const auto& key_shape = key_layout.get_partial_shape();
        if (key_shape.size() != 4 || !key_shape[2].is_static() || !key_shape[3].is_static()) {
            return true; // force to cm impl
        }
        const size_t shape2 = static_cast<size_t>(key_shape[2].get_length());
        const size_t shape3 = static_cast<size_t>(key_shape[3].get_length());
        const bool token_major = shape2 >= shape3;  // CM places BLOCK_SIZE (>=128 typically) in shape[2]
        return token_major;
    }
};

}  // namespace ov::intel_gpu::cm
