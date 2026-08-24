// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_catalog.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../eltwise_shader_abi.hpp"
#include "eltwise_broadcast_f32_eq_spirv.hpp"
#include "eltwise_broadcast_fast_f32_eq_spirv.hpp"
#include "eltwise_broadcast_fast_spirv.hpp"
#include "eltwise_broadcast_fast_vector_spirv.hpp"
#include "eltwise_broadcast_vector_spirv.hpp"
#include "eltwise_dense_f32_div_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_div_push_constants_spirv.hpp"
#include "eltwise_dense_f32_sum_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_sum_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec2_no_tail_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_vec2_no_tail_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec2_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_vec2_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec4_no_tail_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_vec4_no_tail_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec4_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_f32_vec4_push_constants_spirv.hpp"
#include "eltwise_dense_i64_div_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_i64_div_push_constants_spirv.hpp"
#include "eltwise_dense_i64_sum_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_i64_sum_push_constants_spirv.hpp"
#include "eltwise_dense_packed_16bit_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_packed_16bit_push_constants_spirv.hpp"
#include "eltwise_dense_packed_8bit_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_packed_8bit_push_constants_spirv.hpp"
#include "eltwise_dense_packed_f16_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_packed_f16_push_constants_spirv.hpp"
#include "eltwise_dense_push_constants_restrict_spirv.hpp"
#include "eltwise_dense_push_constants_spirv.hpp"
#include "eltwise_dense_restrict_spirv.hpp"
#include "eltwise_dense_spirv.hpp"
#include "eltwise_fused_broadcast_spirv.hpp"
#include "eltwise_fused_dense_chain_length2_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length2_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length2_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length2_spirv.hpp"
#include "eltwise_fused_dense_chain_length3_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length3_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length3_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length3_spirv.hpp"
#include "eltwise_fused_dense_chain_length4_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length4_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length4_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length4_spirv.hpp"
#include "eltwise_fused_dense_chain_length5_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length5_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length5_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length5_spirv.hpp"
#include "eltwise_fused_dense_chain_length6_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length6_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length6_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length6_spirv.hpp"
#include "eltwise_fused_dense_chain_length7_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length7_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length7_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length7_spirv.hpp"
#include "eltwise_fused_dense_chain_length8_f32_vec4_no_tail_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length8_f32_vec4_no_tail_spirv.hpp"
#include "eltwise_fused_dense_chain_length8_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_length8_spirv.hpp"
#include "eltwise_fused_dense_chain_restrict_spirv.hpp"
#include "eltwise_fused_dense_chain_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_no_tail_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_no_tail_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_no_tail_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_no_tail_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_push_constants_spirv.hpp"
#include "eltwise_fused_dense_packed_16bit_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_packed_16bit_push_constants_spirv.hpp"
#include "eltwise_fused_dense_packed_8bit_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_packed_8bit_push_constants_spirv.hpp"
#include "eltwise_fused_dense_packed_f16_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_packed_f16_push_constants_spirv.hpp"
#include "eltwise_fused_dense_push_constants_restrict_spirv.hpp"
#include "eltwise_fused_dense_push_constants_spirv.hpp"
#include "eltwise_fused_dense_restrict_spirv.hpp"
#include "eltwise_fused_dense_spirv.hpp"
#include "eltwise_fused_post_op_spirv.hpp"
#include "eltwise_scalar_constant_spirv.hpp"
#include "eltwise_spirv.hpp"
#include "eltwise_unary_spirv.hpp"
#include "openvino/core/except.hpp"

namespace cldnn::vulkan::eltwise_detail {
namespace {

constexpr uint32_t min_multi_stage_fused_chain_length = 2;
constexpr uint32_t max_fused_chain_length = eltwise_shader_abi::value(eltwise_shader_abi::limit::max_fused_chain_length);

struct spirv_blob {
    const uint32_t* data = nullptr;
    size_t size = 0;
};

std::shared_ptr<kernel_string> make_kernel_source(kernel_kind kind, bool restrict_output, uint32_t fused_chain_length) {
    OPENVINO_ASSERT(!restrict_output || supports_restricted_output(kind), "[GPU][Vulkan] Restricted Eltwise module requires the single-output dense ABI");
    auto source = std::make_shared<kernel_string>();
    const uint32_t* spirv = nullptr;
    size_t spirv_size = 0;
    const auto select_spirv = [&](const auto& alias_safe, const auto& restricted) {
        if (restrict_output) {
            spirv = restricted;
            spirv_size = sizeof(restricted);
        } else {
            spirv = alias_safe;
            spirv_size = sizeof(alias_safe);
        }
    };
    const auto select_fused_chain_spirv = [&](const auto& alias_safe_variants, const auto& restricted_variants) {
        OPENVINO_ASSERT(fused_chain_length >= min_multi_stage_fused_chain_length && fused_chain_length <= max_fused_chain_length,
                        "[GPU][Vulkan] Invalid fixed Eltwise chain length");
        const auto variant_index = fused_chain_length - min_multi_stage_fused_chain_length;
        const auto& selected = restrict_output ? restricted_variants.at(variant_index) : alias_safe_variants.at(variant_index);
        spirv = selected.data;
        spirv_size = selected.size;
    };
    if (kind == kernel_kind::dense) {
        select_spirv(eltwise_dense_spirv, eltwise_dense_restrict_spirv);
    } else if (kind == kernel_kind::dense_push_constants) {
        select_spirv(eltwise_dense_push_constants_spirv, eltwise_dense_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_sum_push_constants) {
        select_spirv(eltwise_dense_f32_sum_push_constants_spirv, eltwise_dense_f32_sum_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_div_push_constants) {
        select_spirv(eltwise_dense_f32_div_push_constants_spirv, eltwise_dense_f32_div_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_i64_sum_push_constants) {
        select_spirv(eltwise_dense_i64_sum_push_constants_spirv, eltwise_dense_i64_sum_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_i64_div_push_constants) {
        select_spirv(eltwise_dense_i64_div_push_constants_spirv, eltwise_dense_i64_div_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_vec2_push_constants) {
        select_spirv(eltwise_dense_f32_vec2_push_constants_spirv, eltwise_dense_f32_vec2_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_vec4_push_constants) {
        select_spirv(eltwise_dense_f32_vec4_push_constants_spirv, eltwise_dense_f32_vec4_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_vec2_no_tail_push_constants) {
        select_spirv(eltwise_dense_f32_vec2_no_tail_push_constants_spirv, eltwise_dense_f32_vec2_no_tail_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_f32_vec4_no_tail_push_constants) {
        select_spirv(eltwise_dense_f32_vec4_no_tail_push_constants_spirv, eltwise_dense_f32_vec4_no_tail_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_packed_8bit_push_constants) {
        select_spirv(eltwise_dense_packed_8bit_push_constants_spirv, eltwise_dense_packed_8bit_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_packed_16bit_push_constants) {
        select_spirv(eltwise_dense_packed_16bit_push_constants_spirv, eltwise_dense_packed_16bit_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::dense_packed_f16_push_constants) {
        select_spirv(eltwise_dense_packed_f16_push_constants_spirv, eltwise_dense_packed_f16_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense) {
        select_spirv(eltwise_fused_dense_spirv, eltwise_fused_dense_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_push_constants) {
        select_spirv(eltwise_fused_dense_push_constants_spirv, eltwise_fused_dense_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec2_push_constants) {
        select_spirv(eltwise_fused_dense_f32_vec2_push_constants_spirv, eltwise_fused_dense_f32_vec2_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec4_push_constants) {
        select_spirv(eltwise_fused_dense_f32_vec4_push_constants_spirv, eltwise_fused_dense_f32_vec4_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec2_no_tail_push_constants) {
        select_spirv(eltwise_fused_dense_f32_vec2_no_tail_push_constants_spirv, eltwise_fused_dense_f32_vec2_no_tail_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec4_no_tail_push_constants) {
        select_spirv(eltwise_fused_dense_f32_vec4_no_tail_push_constants_spirv, eltwise_fused_dense_f32_vec4_no_tail_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_8bit_push_constants) {
        select_spirv(eltwise_fused_dense_packed_8bit_push_constants_spirv, eltwise_fused_dense_packed_8bit_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_16bit_push_constants) {
        select_spirv(eltwise_fused_dense_packed_16bit_push_constants_spirv, eltwise_fused_dense_packed_16bit_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_f16_push_constants) {
        select_spirv(eltwise_fused_dense_packed_f16_push_constants_spirv, eltwise_fused_dense_packed_f16_push_constants_restrict_spirv);
    } else if (kind == kernel_kind::fused_broadcast) {
        spirv = eltwise_fused_broadcast_spirv;
        spirv_size = sizeof(eltwise_fused_broadcast_spirv);
    } else if (kind == kernel_kind::fused_post_op) {
        spirv = eltwise_fused_post_op_spirv;
        spirv_size = sizeof(eltwise_fused_post_op_spirv);
    } else if (kind == kernel_kind::fused_dense_chain || kind == kernel_kind::fused_dense_chain_f32_vec4_no_tail) {
        constexpr size_t fixed_chain_variant_count = max_fused_chain_length - min_multi_stage_fused_chain_length + 1;
        static const std::array<spirv_blob, fixed_chain_variant_count> scalar_alias_safe{{
            {eltwise_fused_dense_chain_length2_spirv, sizeof(eltwise_fused_dense_chain_length2_spirv)},
            {eltwise_fused_dense_chain_length3_spirv, sizeof(eltwise_fused_dense_chain_length3_spirv)},
            {eltwise_fused_dense_chain_length4_spirv, sizeof(eltwise_fused_dense_chain_length4_spirv)},
            {eltwise_fused_dense_chain_length5_spirv, sizeof(eltwise_fused_dense_chain_length5_spirv)},
            {eltwise_fused_dense_chain_length6_spirv, sizeof(eltwise_fused_dense_chain_length6_spirv)},
            {eltwise_fused_dense_chain_length7_spirv, sizeof(eltwise_fused_dense_chain_length7_spirv)},
            {eltwise_fused_dense_chain_length8_spirv, sizeof(eltwise_fused_dense_chain_length8_spirv)},
        }};
        static const std::array<spirv_blob, fixed_chain_variant_count> scalar_restricted{{
            {eltwise_fused_dense_chain_length2_restrict_spirv, sizeof(eltwise_fused_dense_chain_length2_restrict_spirv)},
            {eltwise_fused_dense_chain_length3_restrict_spirv, sizeof(eltwise_fused_dense_chain_length3_restrict_spirv)},
            {eltwise_fused_dense_chain_length4_restrict_spirv, sizeof(eltwise_fused_dense_chain_length4_restrict_spirv)},
            {eltwise_fused_dense_chain_length5_restrict_spirv, sizeof(eltwise_fused_dense_chain_length5_restrict_spirv)},
            {eltwise_fused_dense_chain_length6_restrict_spirv, sizeof(eltwise_fused_dense_chain_length6_restrict_spirv)},
            {eltwise_fused_dense_chain_length7_restrict_spirv, sizeof(eltwise_fused_dense_chain_length7_restrict_spirv)},
            {eltwise_fused_dense_chain_length8_restrict_spirv, sizeof(eltwise_fused_dense_chain_length8_restrict_spirv)},
        }};
        static const std::array<spirv_blob, fixed_chain_variant_count> vector_alias_safe{{
            {eltwise_fused_dense_chain_length2_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length2_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length3_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length3_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length4_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length4_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length5_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length5_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length6_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length6_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length7_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length7_f32_vec4_no_tail_spirv)},
            {eltwise_fused_dense_chain_length8_f32_vec4_no_tail_spirv, sizeof(eltwise_fused_dense_chain_length8_f32_vec4_no_tail_spirv)},
        }};
        static const std::array<spirv_blob, fixed_chain_variant_count> vector_restricted{{
            {eltwise_fused_dense_chain_length2_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length2_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length3_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length3_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length4_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length4_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length5_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length5_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length6_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length6_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length7_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length7_f32_vec4_no_tail_restrict_spirv)},
            {eltwise_fused_dense_chain_length8_f32_vec4_no_tail_restrict_spirv, sizeof(eltwise_fused_dense_chain_length8_f32_vec4_no_tail_restrict_spirv)},
        }};
        if (kind == kernel_kind::fused_dense_chain_f32_vec4_no_tail) {
            select_fused_chain_spirv(vector_alias_safe, vector_restricted);
        } else {
            select_fused_chain_spirv(scalar_alias_safe, scalar_restricted);
        }
    } else if (kind == kernel_kind::broadcast_vector) {
        spirv = eltwise_broadcast_vector_spirv;
        spirv_size = sizeof(eltwise_broadcast_vector_spirv);
    } else if (kind == kernel_kind::broadcast_fast_scalar) {
        spirv = eltwise_broadcast_fast_spirv;
        spirv_size = sizeof(eltwise_broadcast_fast_spirv);
    } else if (kind == kernel_kind::broadcast_f32_eq) {
        spirv = eltwise_broadcast_f32_eq_spirv;
        spirv_size = sizeof(eltwise_broadcast_f32_eq_spirv);
    } else if (kind == kernel_kind::broadcast_fast_f32_eq) {
        spirv = eltwise_broadcast_fast_f32_eq_spirv;
        spirv_size = sizeof(eltwise_broadcast_fast_f32_eq_spirv);
    } else if (kind == kernel_kind::broadcast_fast_vector) {
        spirv = eltwise_broadcast_fast_vector_spirv;
        spirv_size = sizeof(eltwise_broadcast_fast_vector_spirv);
    } else if (kind == kernel_kind::unary) {
        spirv = eltwise_unary_spirv;
        spirv_size = sizeof(eltwise_unary_spirv);
    } else if (kind == kernel_kind::scalar_constant) {
        spirv = eltwise_scalar_constant_spirv;
        spirv_size = sizeof(eltwise_scalar_constant_spirv);
    } else {
        spirv = eltwise_spirv;
        spirv_size = sizeof(eltwise_spirv);
    }
    source->str.assign(reinterpret_cast<const char*>(spirv), spirv_size);
    source->entry_point = "main";
    source->batch_compilation = false;
    source->language = kernel_language::SPIRV;
    return source;
}

}  // namespace

std::vector<std::shared_ptr<kernel_string>> make_kernel_sources(kernel_kind kind, uint32_t fused_chain_length) {
    std::vector<std::shared_ptr<kernel_string>> sources{make_kernel_source(kind, false, fused_chain_length)};
    if (supports_restricted_output(kind)) {
        sources.push_back(make_kernel_source(kind, true, fused_chain_length));
    }
    return sources;
}

}  // namespace cldnn::vulkan::eltwise_detail
