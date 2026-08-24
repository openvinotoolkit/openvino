// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "../eltwise_shader_abi.hpp"
#include "eltwise_inst.h"
#include "fusion_analysis.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "kernel_kind.hpp"
#include "kernel_selection.hpp"

namespace cldnn::vulkan::eltwise_detail {

class EltwiseMetadata final {
public:
    static constexpr uint32_t maximum_rank = 8;
    static constexpr uint32_t minimum_multi_stage_fused_chain_length = 2;
    static constexpr uint32_t maximum_fused_chain_length = eltwise_shader_abi::value(eltwise_shader_abi::limit::max_fused_chain_length);

    static EltwiseMetadata build(const eltwise_inst& instance,
                                 const std::optional<scalar_constant>& scalar,
                                 const std::optional<fused_eltwise_chain>& fused,
                                 const std::optional<fused_post_op_info>& post_op);

    static uint32_t buffer_word_count(kernel_kind kind);
    static constexpr uint32_t dense_push_constant_bytes(bool fused) {
        return (eltwise_shader_abi::index(eltwise_shader_abi::dense_metadata_field::count) +
                (fused ? eltwise_shader_abi::index(eltwise_shader_abi::fused_dense_metadata_field::count) : 0U)) *
               sizeof(uint32_t);
    }

    static uint32_t collapsed_broadcast_rank(const layout& input0_layout, const layout& input1_layout, const layout& output_layout);

    uint32_t operator[](size_t index) const {
        return _words[index];
    }

    const uint32_t* data() const {
        return _words.data();
    }

    bool operator==(const EltwiseMetadata& other) const {
        return _words == other._words;
    }

    bool operator!=(const EltwiseMetadata& other) const {
        return !(*this == other);
    }

    uint32_t active_broadcast_axes(eltwise_shader_abi::tensor_index tensor) const;
    uint32_t fused_stage_value(size_t stage, eltwise_shader_abi::fused_metadata_field field) const;
    scalars_desc make_dense_push_constants(bool fused) const;

private:
    friend class EltwiseMetadataBuilder;

    static constexpr uint32_t header_words = eltwise_shader_abi::index(eltwise_shader_abi::metadata_field::count);
    static constexpr uint32_t tensor_words = maximum_rank * 2 + 1;
    static constexpr uint32_t tensor_count = eltwise_shader_abi::index(eltwise_shader_abi::tensor_index::count);
    static constexpr uint32_t fused_metadata_base = header_words + tensor_count * tensor_words;
    static constexpr uint32_t fused_metadata_words = eltwise_shader_abi::index(eltwise_shader_abi::fused_metadata_field::count);
    static constexpr uint32_t regular_fast_divisor_metadata_base = fused_metadata_base + fused_metadata_words;
    static constexpr uint32_t regular_metadata_words = regular_fast_divisor_metadata_base + maximum_rank;
    static constexpr uint32_t fused_chain_metadata_base = fused_metadata_base + maximum_fused_chain_length * fused_metadata_words;
    static constexpr uint32_t fused_chain_fast_divisor_metadata_base =
        fused_chain_metadata_base + eltwise_shader_abi::index(eltwise_shader_abi::fused_chain_metadata_field::count);
    static constexpr uint32_t fused_chain_metadata_words = fused_chain_fast_divisor_metadata_base + maximum_rank;
    static constexpr uint32_t fused_broadcast_metadata_base = fused_chain_metadata_words;
    static constexpr uint32_t fused_broadcast_metadata_words = fused_broadcast_metadata_base + tensor_words;
    static constexpr uint32_t post_op_metadata_base = fused_broadcast_metadata_words;
    static constexpr uint32_t post_op_metadata_words = post_op_metadata_base + eltwise_shader_abi::index(eltwise_shader_abi::post_op_metadata_field::count);
    static constexpr uint32_t word_count = post_op_metadata_words;

    using storage_type = std::array<uint32_t, word_count>;

    storage_type _words{};
};

}  // namespace cldnn::vulkan::eltwise_detail
