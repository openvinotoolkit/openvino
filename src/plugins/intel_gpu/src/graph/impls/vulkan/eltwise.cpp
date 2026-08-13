// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "data_inst.h"
#include "eltwise_broadcast_vector_spirv.hpp"
#include "eltwise_dense_f32_vec2_no_tail_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec2_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec4_no_tail_push_constants_spirv.hpp"
#include "eltwise_dense_f32_vec4_push_constants_spirv.hpp"
#include "eltwise_dense_packed_16bit_push_constants_spirv.hpp"
#include "eltwise_dense_packed_8bit_push_constants_spirv.hpp"
#include "eltwise_dense_packed_f16_push_constants_spirv.hpp"
#include "eltwise_dense_push_constants_spirv.hpp"
#include "eltwise_dense_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_no_tail_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec2_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_no_tail_push_constants_spirv.hpp"
#include "eltwise_fused_dense_f32_vec4_push_constants_spirv.hpp"
#include "eltwise_fused_dense_packed_16bit_push_constants_spirv.hpp"
#include "eltwise_fused_dense_packed_8bit_push_constants_spirv.hpp"
#include "eltwise_fused_dense_packed_f16_push_constants_spirv.hpp"
#include "eltwise_fused_dense_push_constants_spirv.hpp"
#include "eltwise_fused_dense_spirv.hpp"
#include "eltwise_scalar_constant_spirv.hpp"
#include "eltwise_shader_abi.hpp"
#include "eltwise_spirv.hpp"
#include "eltwise_unary_spirv.hpp"
#include "impls/ocl/kernels_cache.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"
#include "registry/implementation_map.hpp"
#include "vulkan/vulkan_engine.hpp"

namespace cldnn {
namespace vulkan {
namespace {

namespace shader_abi = eltwise_shader_abi;

constexpr uint32_t max_rank = 8;
constexpr uint32_t header_words = shader_abi::index(shader_abi::metadata_field::count);
constexpr uint32_t tensor_words = max_rank * 2 + 1;
constexpr uint32_t tensor_count = shader_abi::index(shader_abi::tensor_index::count);
constexpr uint32_t fused_metadata_base = header_words + tensor_count * tensor_words;
constexpr uint32_t metadata_words = fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::count);
constexpr uint32_t dense_push_constant_words = shader_abi::index(shader_abi::dense_metadata_field::count);
constexpr uint32_t fused_dense_push_constant_words = dense_push_constant_words + shader_abi::index(shader_abi::fused_dense_metadata_field::count);
constexpr uint32_t dense_push_constant_bytes = dense_push_constant_words * sizeof(uint32_t);
constexpr uint32_t fused_dense_push_constant_bytes = fused_dense_push_constant_words * sizeof(uint32_t);
constexpr uint32_t portable_max_local_work_group_size = 128;
constexpr uint32_t dense_vector_bytes = 16;
constexpr uint32_t broadcast_vector_width = 4;
constexpr uint32_t broadcast_vector_min_elements = 256;

enum class kernel_kind : uint8_t {
    dense,
    broadcast_scalar,
    broadcast_vector,
    unary,
    scalar_constant,
    fused_dense,
    dense_push_constants,
    fused_dense_push_constants,
    dense_f32_vec2_push_constants,
    dense_f32_vec4_push_constants,
    fused_dense_f32_vec2_push_constants,
    fused_dense_f32_vec4_push_constants,
    dense_f32_vec2_no_tail_push_constants,
    dense_f32_vec4_no_tail_push_constants,
    fused_dense_f32_vec2_no_tail_push_constants,
    fused_dense_f32_vec4_no_tail_push_constants,
    dense_packed_8bit_push_constants,
    dense_packed_16bit_push_constants,
    fused_dense_packed_8bit_push_constants,
    fused_dense_packed_16bit_push_constants,
    dense_packed_f16_push_constants,
    fused_dense_packed_f16_push_constants,
};

enum class dense_vector_width : uint32_t {
    scalar = 1,
    vec2 = 2,
    vec4 = 4,
};

enum class packed_dense_width : uint32_t {
    none = 0,
    two_16bit_elements = 2,
    four_8bit_elements = 4,
};

constexpr uint32_t value(dense_vector_width width) {
    return static_cast<uint32_t>(width);
}

bool is_f32_vector_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_f32_vec2_push_constants,
                   kernel_kind::dense_f32_vec4_push_constants,
                   kernel_kind::fused_dense_f32_vec2_push_constants,
                   kernel_kind::fused_dense_f32_vec4_push_constants,
                   kernel_kind::dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec4_no_tail_push_constants});
}

bool is_packed_dense_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense_packed_8bit_push_constants,
                   kernel_kind::dense_packed_16bit_push_constants,
                   kernel_kind::fused_dense_packed_8bit_push_constants,
                   kernel_kind::fused_dense_packed_16bit_push_constants,
                   kernel_kind::dense_packed_f16_push_constants,
                   kernel_kind::fused_dense_packed_f16_push_constants});
}

bool uses_dense_push_constants(kernel_kind kind) {
    return kind == kernel_kind::dense_push_constants || kind == kernel_kind::fused_dense_push_constants || is_f32_vector_kernel(kind) ||
           is_packed_dense_kernel(kind);
}

bool is_plain_dense_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::dense,
                   kernel_kind::dense_push_constants,
                   kernel_kind::dense_f32_vec2_push_constants,
                   kernel_kind::dense_f32_vec4_push_constants,
                   kernel_kind::dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::dense_packed_8bit_push_constants,
                   kernel_kind::dense_packed_16bit_push_constants,
                   kernel_kind::dense_packed_f16_push_constants});
}

bool is_fused_dense_kernel(kernel_kind kind) {
    return one_of(kind,
                  {kernel_kind::fused_dense,
                   kernel_kind::fused_dense_push_constants,
                   kernel_kind::fused_dense_f32_vec2_push_constants,
                   kernel_kind::fused_dense_f32_vec4_push_constants,
                   kernel_kind::fused_dense_f32_vec2_no_tail_push_constants,
                   kernel_kind::fused_dense_f32_vec4_no_tail_push_constants,
                   kernel_kind::fused_dense_packed_8bit_push_constants,
                   kernel_kind::fused_dense_packed_16bit_push_constants,
                   kernel_kind::fused_dense_packed_f16_push_constants});
}

dense_vector_width get_dense_vector_width(kernel_kind kind) {
    if (one_of(kind,
               {kernel_kind::dense_f32_vec4_push_constants,
                kernel_kind::fused_dense_f32_vec4_push_constants,
                kernel_kind::dense_f32_vec4_no_tail_push_constants,
                kernel_kind::fused_dense_f32_vec4_no_tail_push_constants})) {
        return dense_vector_width::vec4;
    }
    if (one_of(kind,
               {kernel_kind::dense_f32_vec2_push_constants,
                kernel_kind::fused_dense_f32_vec2_push_constants,
                kernel_kind::dense_f32_vec2_no_tail_push_constants,
                kernel_kind::fused_dense_f32_vec2_no_tail_push_constants})) {
        return dense_vector_width::vec2;
    }
    return dense_vector_width::scalar;
}

packed_dense_width get_packed_dense_width(kernel_kind kind) {
    if (one_of(kind, {kernel_kind::dense_packed_8bit_push_constants, kernel_kind::fused_dense_packed_8bit_push_constants})) {
        return packed_dense_width::four_8bit_elements;
    }
    if (one_of(kind,
               {kernel_kind::dense_packed_16bit_push_constants,
                kernel_kind::fused_dense_packed_16bit_push_constants,
                kernel_kind::dense_packed_f16_push_constants,
                kernel_kind::fused_dense_packed_f16_push_constants})) {
        return packed_dense_width::two_16bit_elements;
    }
    return packed_dense_width::none;
}

constexpr uint32_t value(packed_dense_width width) {
    return static_cast<uint32_t>(width);
}

struct scalar_constant {
    shader_abi::tensor_index input_index = shader_abi::tensor_index::input1;
    std::array<uint32_t, 2> bits{};
};

bool is_supported_mode(eltwise_mode mode) {
    return one_of(mode, {eltwise_mode::sum,        eltwise_mode::sub,         eltwise_mode::max,          eltwise_mode::prod,       eltwise_mode::div,
                         eltwise_mode::min,        eltwise_mode::pow,         eltwise_mode::squared_diff, eltwise_mode::mod,        eltwise_mode::eq,
                         eltwise_mode::ne,         eltwise_mode::lt,          eltwise_mode::le,           eltwise_mode::gt,         eltwise_mode::ge,
                         eltwise_mode::logic_and,  eltwise_mode::logic_or,    eltwise_mode::logic_xor,    eltwise_mode::floor_mod,  eltwise_mode::is_finite,
                         eltwise_mode::is_inf,     eltwise_mode::is_nan,      eltwise_mode::right_shift,  eltwise_mode::left_shift, eltwise_mode::bitwise_and,
                         eltwise_mode::bitwise_or, eltwise_mode::bitwise_xor, eltwise_mode::atan2});
}

bool is_unary_mode(eltwise_mode mode) {
    return one_of(mode, {eltwise_mode::is_finite, eltwise_mode::is_inf, eltwise_mode::is_nan});
}

bool is_fused_mode(eltwise_mode mode) {
    return one_of(mode, {eltwise_mode::sum, eltwise_mode::prod, eltwise_mode::sub, eltwise_mode::div});
}

bool is_bitwise_mode(eltwise_mode mode) {
    return one_of(mode, {eltwise_mode::right_shift, eltwise_mode::left_shift, eltwise_mode::bitwise_and, eltwise_mode::bitwise_or, eltwise_mode::bitwise_xor});
}

bool is_supported_data_type(data_types type) {
    return one_of(type,
                  {data_types::f16,
                   data_types::f32,
                   data_types::i8,
                   data_types::u8,
                   data_types::i16,
                   data_types::u16,
                   data_types::i32,
                   data_types::u32,
                   data_types::i64,
                   data_types::boolean});
}

bool is_integer_data_type(data_types type) {
    return one_of(type,
                  {data_types::i8, data_types::u8, data_types::i16, data_types::u16, data_types::i32, data_types::u32, data_types::i64, data_types::boolean});
}

shader_abi::mode shader_mode_code(eltwise_mode mode) {
    switch (mode) {
    case eltwise_mode::sum:
        return shader_abi::mode::sum;
    case eltwise_mode::sub:
        return shader_abi::mode::sub;
    case eltwise_mode::max:
        return shader_abi::mode::max;
    case eltwise_mode::prod:
        return shader_abi::mode::prod;
    case eltwise_mode::div:
        return shader_abi::mode::div;
    case eltwise_mode::min:
        return shader_abi::mode::min;
    case eltwise_mode::pow:
        return shader_abi::mode::pow;
    case eltwise_mode::squared_diff:
        return shader_abi::mode::squared_diff;
    case eltwise_mode::mod:
        return shader_abi::mode::mod;
    case eltwise_mode::eq:
        return shader_abi::mode::eq;
    case eltwise_mode::ne:
        return shader_abi::mode::ne;
    case eltwise_mode::lt:
        return shader_abi::mode::lt;
    case eltwise_mode::le:
        return shader_abi::mode::le;
    case eltwise_mode::gt:
        return shader_abi::mode::gt;
    case eltwise_mode::ge:
        return shader_abi::mode::ge;
    case eltwise_mode::logic_and:
        return shader_abi::mode::logic_and;
    case eltwise_mode::logic_or:
        return shader_abi::mode::logic_or;
    case eltwise_mode::logic_xor:
        return shader_abi::mode::logic_xor;
    case eltwise_mode::floor_mod:
        return shader_abi::mode::floor_mod;
    case eltwise_mode::is_finite:
        return shader_abi::mode::is_finite;
    case eltwise_mode::is_inf:
        return shader_abi::mode::is_inf;
    case eltwise_mode::is_nan:
        return shader_abi::mode::is_nan;
    case eltwise_mode::right_shift:
        return shader_abi::mode::right_shift;
    case eltwise_mode::left_shift:
        return shader_abi::mode::left_shift;
    case eltwise_mode::bitwise_and:
        return shader_abi::mode::bitwise_and;
    case eltwise_mode::bitwise_or:
        return shader_abi::mode::bitwise_or;
    case eltwise_mode::bitwise_xor:
        return shader_abi::mode::bitwise_xor;
    case eltwise_mode::atan2:
        return shader_abi::mode::atan2;
    default:
        OPENVINO_THROW("[GPU][Vulkan] Unsupported Eltwise shader mode");
    }
}

shader_abi::scalar_type scalar_type_code(data_types type) {
    switch (type) {
    case data_types::f16:
        return shader_abi::scalar_type::f16;
    case data_types::f32:
        return shader_abi::scalar_type::f32;
    case data_types::i8:
        return shader_abi::scalar_type::i8;
    case data_types::u8:
        return shader_abi::scalar_type::u8;
    case data_types::i16:
        return shader_abi::scalar_type::i16;
    case data_types::u16:
        return shader_abi::scalar_type::u16;
    case data_types::i32:
        return shader_abi::scalar_type::i32;
    case data_types::u32:
        return shader_abi::scalar_type::u32;
    case data_types::i64:
        return shader_abi::scalar_type::i64;
    case data_types::boolean:
        return shader_abi::scalar_type::boolean;
    default:
        OPENVINO_THROW("[GPU][Vulkan] Unsupported Eltwise scalar type ", ov::element::Type(type).get_type_name());
    }
}

uint32_t float_bits(float value) {
    uint32_t result = 0;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool is_supported_format(format::type fmt) {
    return one_of(fmt, {format::any, format::bfyx, format::yxfb, format::bfzyx, format::bfwzyx, format::bfuwzyx, format::bfvuwzyx});
}

uint32_t checked_u32(size_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] Eltwise ", description, " exceeds the 32-bit shader metadata range");
    return static_cast<uint32_t>(value);
}

uint32_t select_local_work_group_size(uint32_t element_count, uint64_t device_max_work_group_size) {
    const auto limit = static_cast<uint32_t>(std::min<uint64_t>(portable_max_local_work_group_size, device_max_work_group_size));
    OPENVINO_ASSERT(limit > 0, "[GPU][Vulkan] Device reports a zero maximum work-group size");

    uint32_t local_size = 1;
    while (local_size < element_count && local_size <= limit / 2) {
        local_size *= 2;
    }
    return local_size;
}

bool has_dense_storage(const layout& tensor_layout) {
    return !static_cast<bool>(tensor_layout.data_padding) &&
           tensor_layout.bytes_count() == tensor_layout.count() * data_type_traits::size_of(tensor_layout.data_type);
}

std::optional<scalar_constant> get_scalar_constant(const program_node& node) {
    for (uint32_t input_index = shader_abi::index(shader_abi::tensor_index::input0); input_index <= shader_abi::index(shader_abi::tensor_index::input1);
         ++input_index) {
        const auto& dependency = node.get_dependency(input_index);
        const auto& input_layout = dependency.get_output_layout(false);
        if (!dependency.is_type<data>() || !dependency.is_constant() || input_layout.is_dynamic() || input_layout.count() != 1 ||
            input_layout.get_linear_offset() != 0 || !has_dense_storage(input_layout)) {
            continue;
        }

        scalar_constant result;
        result.input_index = static_cast<shader_abi::tensor_index>(input_index);
        const auto value_size = data_type_traits::size_of(input_layout.data_type);
        OPENVINO_ASSERT(value_size <= sizeof(result.bits), "[GPU][Vulkan] Eltwise scalar constant exceeds the supported storage size");
        mem_lock<uint8_t, mem_lock_type::read> constant_data{dependency.as<data>().get_attached_memory_ptr(), node.get_program().get_stream()};
        std::memcpy(result.bits.data(), constant_data.data(), value_size);
        return result;
    }
    return std::nullopt;
}

bool can_use_scalar_linear_storage(const layout& tensor_layout, const layout& output_layout) {
    return has_dense_storage(tensor_layout) && has_dense_storage(output_layout) && tensor_layout.count() == output_layout.count();
}

bool can_use_linear_storage(const layout& input0_layout, const layout& input1_layout, const layout& output_layout) {
    return has_dense_storage(input0_layout) && has_dense_storage(input1_layout) && has_dense_storage(output_layout) && input0_layout.identical(output_layout) &&
           input1_layout.identical(output_layout);
}

uint32_t output_elements_per_invocation(const layout& output_layout) {
    const auto scalar_size = data_type_traits::size_of(output_layout.data_type);
    if (scalar_size >= sizeof(uint32_t) || !has_dense_storage(output_layout)) {
        return 1;
    }
    OPENVINO_ASSERT(sizeof(uint32_t) % scalar_size == 0, "[GPU][Vulkan] Eltwise output scalar size cannot be packed into a 32-bit storage word");
    return checked_u32(sizeof(uint32_t) / scalar_size, "packed output width");
}

uint32_t dense_elements_per_invocation(const layout& output_layout) {
    const auto scalar_size = data_type_traits::size_of(output_layout.data_type);
    return checked_u32(std::max<size_t>(1, dense_vector_bytes / scalar_size), "dense vector width");
}

uint32_t scalar_constant_elements_per_invocation(const layout& output_layout) {
    return data_type_traits::size_of(output_layout.data_type) >= sizeof(uint32_t) ? broadcast_vector_width : output_elements_per_invocation(output_layout);
}

uint32_t fused_dense_elements_per_invocation(const layout& output_layout) {
    return data_type_traits::size_of(output_layout.data_type) >= sizeof(uint32_t) ? dense_elements_per_invocation(output_layout)
                                                                                  : output_elements_per_invocation(output_layout);
}

bool can_use_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& output_layout) {
    return data_type_traits::size_of(output_layout.data_type) >= sizeof(uint32_t) && can_use_linear_storage(input0_layout, input1_layout, output_layout);
}

packed_dense_width packed_width_for_type(data_types type) {
    if (one_of(type, {data_types::i8, data_types::u8})) {
        return packed_dense_width::four_8bit_elements;
    }
    if (one_of(type, {data_types::f16, data_types::i16, data_types::u16})) {
        return packed_dense_width::two_16bit_elements;
    }
    return packed_dense_width::none;
}

bool is_aligned_for_packed_width(const layout& tensor_layout, packed_dense_width width) {
    return tensor_layout.get_linear_offset() % value(width) == 0;
}

bool can_use_packed_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& output_layout, const layout* fused_input_layout) {
    const auto width = packed_width_for_type(output_layout.data_type);
    if (width == packed_dense_width::none || input0_layout.data_type != output_layout.data_type || input1_layout.data_type != output_layout.data_type ||
        !can_use_linear_storage(input0_layout, input1_layout, output_layout) || output_layout.count() % value(width) != 0 ||
        !is_aligned_for_packed_width(input0_layout, width) || !is_aligned_for_packed_width(input1_layout, width) ||
        !is_aligned_for_packed_width(output_layout, width)) {
        return false;
    }
    return fused_input_layout == nullptr || (fused_input_layout->data_type == output_layout.data_type && fused_input_layout->identical(output_layout) &&
                                             has_dense_storage(*fused_input_layout) && is_aligned_for_packed_width(*fused_input_layout, width));
}

packed_dense_width select_packed_dense_width(const layout& input0_layout,
                                             const layout& input1_layout,
                                             const layout& output_layout,
                                             const layout* fused_input_layout,
                                             const device_info& info) {
    if (!can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, fused_input_layout) || info.supported_simd_sizes.empty() ||
        info.supported_simd_sizes.front() == 0) {
        return packed_dense_width::none;
    }
    const auto width = packed_width_for_type(output_layout.data_type);
    const auto logical_elements_per_subgroup = static_cast<uint64_t>(info.supported_simd_sizes.front()) * value(width);
    if (logical_elements_per_subgroup > portable_max_local_work_group_size) {
        return packed_dense_width::none;
    }
    const auto invocation_count = output_layout.count() / value(width);
    const auto useful_subgroup_width = std::min<uint64_t>(std::max<uint64_t>(info.max_work_group_size, 1), info.supported_simd_sizes.front());
    return invocation_count >= useful_subgroup_width ? width : packed_dense_width::none;
}

kernel_kind select_packed_dense_kernel_kind(packed_dense_width width, data_types type, bool fused) {
    if (width == packed_dense_width::four_8bit_elements) {
        return fused ? kernel_kind::fused_dense_packed_8bit_push_constants : kernel_kind::dense_packed_8bit_push_constants;
    }
    OPENVINO_ASSERT(width == packed_dense_width::two_16bit_elements, "[GPU][Vulkan] Invalid packed dense width selected for Eltwise");
    if (type == data_types::f16) {
        return fused ? kernel_kind::fused_dense_packed_f16_push_constants : kernel_kind::dense_packed_f16_push_constants;
    }
    return fused ? kernel_kind::fused_dense_packed_16bit_push_constants : kernel_kind::dense_packed_16bit_push_constants;
}

bool can_use_fused_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& fused_input_layout, const layout& output_layout) {
    if (!can_use_linear_storage(input0_layout, input1_layout, output_layout) || !has_dense_storage(fused_input_layout) ||
        !fused_input_layout.identical(output_layout)) {
        return false;
    }
    const auto packed_width = output_elements_per_invocation(output_layout);
    return packed_width == 1 || (output_layout.count() % packed_width == 0 && output_layout.get_linear_offset() % packed_width == 0);
}

bool is_aligned_for_vector_width(const layout& tensor_layout, dense_vector_width width) {
    return tensor_layout.get_linear_offset() % value(width) == 0;
}

bool can_use_f32_dense_vector_width(const layout& input0_layout,
                                    const layout& input1_layout,
                                    const layout& output_layout,
                                    const layout* fused_input_layout,
                                    dense_vector_width width) {
    return input0_layout.data_type == data_types::f32 && input1_layout.data_type == data_types::f32 && output_layout.data_type == data_types::f32 &&
           (fused_input_layout == nullptr || fused_input_layout->data_type == data_types::f32) && is_aligned_for_vector_width(input0_layout, width) &&
           is_aligned_for_vector_width(input1_layout, width) && is_aligned_for_vector_width(output_layout, width) &&
           (fused_input_layout == nullptr || is_aligned_for_vector_width(*fused_input_layout, width));
}

dense_vector_width select_f32_dense_vector_width(const layout& input0_layout,
                                                 const layout& input1_layout,
                                                 const layout& output_layout,
                                                 const layout* fused_input_layout,
                                                 const device_info& info) {
    const auto max_work_group_size = std::max<uint64_t>(info.max_work_group_size, 1);
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0) {
        return dense_vector_width::scalar;
    }
    const auto subgroup_size = static_cast<uint64_t>(info.supported_simd_sizes.front());
    const auto subgroups_per_full_work_group = (max_work_group_size + subgroup_size - 1) / subgroup_size;
    constexpr std::array candidates{dense_vector_width::vec4, dense_vector_width::vec2};
    for (const auto width : candidates) {
        if (!can_use_f32_dense_vector_width(input0_layout, input1_layout, output_layout, fused_input_layout, width)) {
            continue;
        }
        const auto vector_width = value(width);
        const auto element_count = output_layout.count();
        const auto invocation_count = element_count / vector_width + (element_count % vector_width != 0 ? 1 : 0);
        const bool wide_subgroup = subgroup_size >= subgroups_per_full_work_group * vector_width;
        const auto required_full_work_groups = wide_subgroup ? uint64_t{1} : subgroups_per_full_work_group;
        if (invocation_count / max_work_group_size >= required_full_work_groups) {
            return width;
        }
    }
    return dense_vector_width::scalar;
}

bool can_use_f32_no_tail_kernel(const layout& output_layout, dense_vector_width width, const device_info& info) {
    const auto vector_width = value(width);
    if (output_layout.count() % vector_width != 0 || info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0) {
        return false;
    }
    const auto invocation_count = checked_u32(output_layout.count() / vector_width, "F32 vector invocation count");
    const auto max_work_group_size = std::max<uint64_t>(info.max_work_group_size, 1);
    const auto local_size = select_local_work_group_size(invocation_count, max_work_group_size);
    if (invocation_count % local_size != 0) {
        return false;
    }

    const auto subgroup_size = static_cast<uint64_t>(info.supported_simd_sizes.front());
    const auto subgroups_per_full_work_group = (max_work_group_size + subgroup_size - 1) / subgroup_size;
    const bool wide_subgroup = subgroup_size >= subgroups_per_full_work_group * vector_width;
    const auto full_work_groups = invocation_count / max_work_group_size;
    const auto wide_subgroup_work_group_limit = subgroups_per_full_work_group * subgroups_per_full_work_group;
    return !wide_subgroup || full_work_groups <= wide_subgroup_work_group_limit;
}

kernel_kind select_f32_vector_kernel_kind(dense_vector_width width, bool fused, bool no_tail) {
    if (width == dense_vector_width::vec4) {
        if (no_tail) {
            return fused ? kernel_kind::fused_dense_f32_vec4_no_tail_push_constants : kernel_kind::dense_f32_vec4_no_tail_push_constants;
        }
        return fused ? kernel_kind::fused_dense_f32_vec4_push_constants : kernel_kind::dense_f32_vec4_push_constants;
    }
    OPENVINO_ASSERT(width == dense_vector_width::vec2, "[GPU][Vulkan] Invalid dense vector width selected for Eltwise");
    if (no_tail) {
        return fused ? kernel_kind::fused_dense_f32_vec2_no_tail_push_constants : kernel_kind::dense_f32_vec2_no_tail_push_constants;
    }
    return fused ? kernel_kind::fused_dense_f32_vec2_push_constants : kernel_kind::dense_f32_vec2_push_constants;
}

struct fused_eltwise_info {
    const fused_primitive_desc* descriptor = nullptr;
    size_t external_dependency_index = 0;
    shader_abi::fused_input_position external_position = shader_abi::fused_input_position::rhs;
};

std::optional<fused_eltwise_info> get_supported_fused_eltwise(const program_node& node) {
    const auto& fused_primitives = node.get_fused_primitives();
    if (fused_primitives.size() != 1 || !fused_primitives.front().is_type<eltwise>()) {
        return std::nullopt;
    }

    const auto& fused = fused_primitives.front();
    const auto fused_desc = fused.typed_desc<eltwise>();
    const bool coefficients_supported = fused_desc->coefficients.empty() || (fused_desc->mode == eltwise_mode::sum && fused_desc->coefficients.size() == 2);
    if (!is_fused_mode(fused_desc->mode) || !fused_desc->stride.empty() || !coefficients_supported || fused.inputs.size() != 2 || fused.total_num_deps != 2 ||
        fused.deps.size() != 1 || !fused.has_outer_dep()) {
        return std::nullopt;
    }

    std::optional<size_t> external_position;
    for (size_t position = 0; position < fused.inputs.size(); ++position) {
        const auto& input = fused.inputs[position];
        if (input.m_type == FusedInputType::EXTERNAL) {
            if (external_position.has_value() || input.m_idx >= node.get_dependencies().size()) {
                return std::nullopt;
            }
            external_position = position;
        } else if (input.m_type != FusedInputType::ORIGINAL) {
            return std::nullopt;
        }
    }
    if (!external_position.has_value()) {
        return std::nullopt;
    }

    const auto external_dependency_index = fused.inputs[*external_position].m_idx;
    if (external_dependency_index != static_cast<size_t>(fused.outer_dep_start_idx) || node.get_dependencies().size() != 3) {
        return std::nullopt;
    }

    const auto& output_layout = node.get_output_layout(0);
    if (output_layout.is_dynamic() || !has_dense_storage(output_layout)) {
        return std::nullopt;
    }
    for (size_t input_index : {size_t{0}, size_t{1}, external_dependency_index}) {
        const auto& input_layout = node.get_input_layout(input_index);
        if (input_layout.is_dynamic() || !input_layout.identical(output_layout) || !has_dense_storage(input_layout)) {
            return std::nullopt;
        }
    }
    if (!can_use_fused_dense_kernel(node.get_input_layout(0), node.get_input_layout(1), node.get_input_layout(external_dependency_index), output_layout)) {
        return std::nullopt;
    }

    return fused_eltwise_info{&fused,
                              external_dependency_index,
                              *external_position == 0 ? shader_abi::fused_input_position::lhs : shader_abi::fused_input_position::rhs};
}

bool can_use_broadcast_vector_kernel(const layout& output_layout) {
    if (data_type_traits::size_of(output_layout.data_type) < sizeof(uint32_t) || output_layout.count() < broadcast_vector_min_elements) {
        return false;
    }
    return output_layout.count() % broadcast_vector_width == 0;
}

void write_tensor_metadata(std::array<uint32_t, metadata_words>& metadata, shader_abi::tensor_index tensor, const layout& tensor_layout, uint32_t output_rank) {
    const auto shape = tensor_layout.get_shape();
    const auto pitches = tensor_layout.get_pitches();
    OPENVINO_ASSERT(shape.size() <= pitches.size() && shape.size() <= output_rank, "[GPU][Vulkan] Eltwise received an invalid tensor rank");

    const auto base = header_words + shader_abi::index(tensor) * tensor_words;
    const auto leading_dimensions = output_rank - checked_u32(shape.size(), "rank");
    for (uint32_t axis = 0; axis < max_rank; ++axis) {
        metadata[base + axis] = 1;
        metadata[base + max_rank + axis] = 0;
    }
    for (uint32_t axis = 0; axis < shape.size(); ++axis) {
        const auto output_axis = leading_dimensions + axis;
        metadata[base + output_axis] = checked_u32(shape[axis], "dimension");
        metadata[base + max_rank + output_axis] = checked_u32(pitches[axis], "pitch");
    }
    metadata[base + max_rank * 2] = checked_u32(tensor_layout.get_linear_offset(), "base offset");
}

bool collapsed_tensor_axis(const std::array<uint32_t, metadata_words>& metadata,
                           shader_abi::tensor_index tensor,
                           uint32_t axis,
                           uint32_t output_left_dimension,
                           uint32_t output_right_dimension,
                           uint32_t& collapsed_dimension,
                           uint32_t& collapsed_pitch) {
    const auto base = header_words + shader_abi::index(tensor) * tensor_words;
    const auto left_dimension = metadata[base + axis];
    const auto right_dimension = metadata[base + axis + 1];
    const auto left_pitch = metadata[base + max_rank + axis];
    const auto right_pitch = metadata[base + max_rank + axis + 1];

    if (output_left_dimension == 1) {
        collapsed_dimension = right_dimension;
        collapsed_pitch = right_pitch;
        return true;
    }
    if (output_right_dimension == 1) {
        collapsed_dimension = left_dimension;
        collapsed_pitch = left_pitch;
        return true;
    }
    if (left_dimension == 1 && right_dimension == 1) {
        collapsed_dimension = 1;
        collapsed_pitch = 0;
        return true;
    }
    if (left_dimension == output_left_dimension && right_dimension == output_right_dimension &&
        static_cast<uint64_t>(left_pitch) == static_cast<uint64_t>(right_pitch) * right_dimension) {
        collapsed_dimension = checked_u32(static_cast<uint64_t>(left_dimension) * right_dimension, "collapsed dimension");
        collapsed_pitch = right_pitch;
        return true;
    }
    return false;
}

uint32_t collapse_metadata_dimensions(std::array<uint32_t, metadata_words>& metadata, uint32_t rank) {
    if (rank < 2) {
        return rank;
    }

    for (int32_t axis = static_cast<int32_t>(rank) - 2; axis >= 0; --axis) {
        const auto output_base = header_words + shader_abi::index(shader_abi::tensor_index::output) * tensor_words;
        const auto output_left_dimension = metadata[output_base + static_cast<uint32_t>(axis)];
        const auto output_right_dimension = metadata[output_base + static_cast<uint32_t>(axis) + 1];
        std::array<uint32_t, tensor_count> collapsed_dimensions{};
        std::array<uint32_t, tensor_count> collapsed_pitches{};
        bool can_collapse = true;
        for (uint32_t tensor = 0; tensor < tensor_count; ++tensor) {
            can_collapse &= collapsed_tensor_axis(metadata,
                                                  static_cast<shader_abi::tensor_index>(tensor),
                                                  static_cast<uint32_t>(axis),
                                                  output_left_dimension,
                                                  output_right_dimension,
                                                  collapsed_dimensions[tensor],
                                                  collapsed_pitches[tensor]);
        }
        if (!can_collapse) {
            continue;
        }

        for (uint32_t tensor = 0; tensor < tensor_count; ++tensor) {
            const auto base = header_words + tensor * tensor_words;
            metadata[base + static_cast<uint32_t>(axis)] = collapsed_dimensions[tensor];
            metadata[base + max_rank + static_cast<uint32_t>(axis)] = collapsed_pitches[tensor];
            for (uint32_t shifted_axis = static_cast<uint32_t>(axis) + 1; shifted_axis + 1 < rank; ++shifted_axis) {
                metadata[base + shifted_axis] = metadata[base + shifted_axis + 1];
                metadata[base + max_rank + shifted_axis] = metadata[base + max_rank + shifted_axis + 1];
            }
            metadata[base + rank - 1] = 1;
            metadata[base + max_rank + rank - 1] = 0;
        }
        --rank;
    }
    return rank;
}

std::array<uint32_t, metadata_words> make_metadata(const eltwise_inst& instance,
                                                   const std::optional<scalar_constant>& scalar,
                                                   const std::optional<fused_eltwise_info>& fused) {
    const auto desc = instance.get_typed_desc<eltwise>();
    const auto base_input_count = is_unary_mode(desc->mode) ? 1U : 2U;
    OPENVINO_ASSERT(instance.inputs_memory_count() == base_input_count && instance.get_fused_mem_count() == (fused.has_value() ? 1U : 0U),
                    "[GPU][Vulkan] Eltwise received an unexpected regular or fused input count");
    const auto& input0_layout = instance.get_input_layout(0);
    const auto& input1_layout = base_input_count == 1 ? input0_layout : instance.get_input_layout(1);
    const auto& output_layout = instance.get_output_layout(0);
    OPENVINO_ASSERT(!input0_layout.is_dynamic() && !input1_layout.is_dynamic() && !output_layout.is_dynamic(),
                    "[GPU][Vulkan] Eltwise execution requires resolved runtime layouts");
    const auto output_rank = checked_u32(output_layout.get_shape().size(), "output rank");
    OPENVINO_ASSERT(output_rank <= max_rank, "[GPU][Vulkan] Eltwise supports tensors with rank up to ", max_rank);

    std::array<uint32_t, metadata_words> metadata{};
    metadata[shader_abi::index(shader_abi::metadata_field::element_count)] = checked_u32(output_layout.count(), "element count");
    metadata[shader_abi::index(shader_abi::metadata_field::mode)] = shader_abi::value(shader_mode_code(desc->mode));
    metadata[shader_abi::index(shader_abi::metadata_field::rank)] = output_rank;
    const auto scalar_tensor_index =
        scalar.has_value() && scalar->input_index == shader_abi::tensor_index::input0 ? shader_abi::tensor_index::input1 : shader_abi::tensor_index::input0;
    const auto& scalar_tensor_layout = scalar_tensor_index == shader_abi::tensor_index::input0 ? input0_layout : input1_layout;
    metadata[shader_abi::index(shader_abi::metadata_field::flags)] =
        ((scalar.has_value() ? can_use_scalar_linear_storage(scalar_tensor_layout, output_layout)
                             : can_use_linear_storage(input0_layout, input1_layout, output_layout))
             ? shader_abi::value(shader_abi::storage_flag::linear)
             : 0U) |
        (output_elements_per_invocation(output_layout) > 1 ? shader_abi::value(shader_abi::storage_flag::packed_output) : 0U) |
        (scalar.has_value() && scalar->input_index == shader_abi::tensor_index::input0 ? shader_abi::value(shader_abi::storage_flag::scalar_constant_input0)
                                                                                       : 0U);
    metadata[shader_abi::index(shader_abi::metadata_field::input0_type)] = shader_abi::value(scalar_type_code(input0_layout.data_type));
    metadata[shader_abi::index(shader_abi::metadata_field::input1_type)] = shader_abi::value(scalar_type_code(input1_layout.data_type));
    metadata[shader_abi::index(shader_abi::metadata_field::output_type)] = shader_abi::value(scalar_type_code(output_layout.data_type));
    metadata[shader_abi::index(shader_abi::metadata_field::python_division)] = desc->m_pythondiv ? 1U : 0U;
    metadata[shader_abi::index(shader_abi::metadata_field::infinity_detection)] =
        desc->mode == eltwise_mode::is_inf && desc->coefficients.size() == 2
            ? (desc->coefficients[0] != 0.0f ? shader_abi::value(shader_abi::infinity_flag::negative) : 0U) |
                  (desc->coefficients[1] != 0.0f ? shader_abi::value(shader_abi::infinity_flag::positive) : 0U)
            : shader_abi::all_infinities_mask;
    metadata[shader_abi::index(shader_abi::metadata_field::input_count)] = base_input_count;
    metadata[shader_abi::index(shader_abi::metadata_field::input0_coefficient)] = float_bits(desc->coefficients.empty() ? 1.0f : desc->coefficients[0]);
    metadata[shader_abi::index(shader_abi::metadata_field::input1_coefficient)] = float_bits(desc->coefficients.empty() ? 1.0f : desc->coefficients[1]);
    write_tensor_metadata(metadata, shader_abi::tensor_index::input0, input0_layout, output_rank);
    write_tensor_metadata(metadata, shader_abi::tensor_index::input1, input1_layout, output_rank);
    write_tensor_metadata(metadata, shader_abi::tensor_index::output, output_layout, output_rank);
    metadata[shader_abi::index(shader_abi::metadata_field::rank)] = collapse_metadata_dimensions(metadata, output_rank);
    if (scalar.has_value()) {
        const auto scalar_metadata_base = header_words + shader_abi::index(scalar->input_index) * tensor_words;
        metadata[scalar_metadata_base] = scalar->bits[0];
        metadata[scalar_metadata_base + max_rank] = scalar->bits[1];
    }
    if (fused.has_value()) {
        const auto fused_desc = fused->descriptor->typed_desc<eltwise>();
        const auto& fused_input_layout = instance.get_input_layout(fused->external_dependency_index);
        metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::mode)] = shader_abi::value(shader_mode_code(fused_desc->mode));
        metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_type)] =
            shader_abi::value(scalar_type_code(fused_input_layout.data_type));
        metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_position)] = shader_abi::value(fused->external_position);
        metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::python_division)] = fused_desc->m_pythondiv ? 1U : 0U;
        metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input0_coefficient)] =
            float_bits(fused_desc->coefficients.empty() ? 1.0f : fused_desc->coefficients[0]);
        metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input1_coefficient)] =
            float_bits(fused_desc->coefficients.empty() ? 1.0f : fused_desc->coefficients[1]);
        metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_offset)] =
            checked_u32(fused_input_layout.get_linear_offset(), "fused input base offset");
    }
    return metadata;
}

void append_u32_scalar(scalars_desc& scalars, uint32_t value) {
    scalar_desc scalar{};
    scalar.t = scalar_desc::Types::UINT32;
    scalar.v.u32 = value;
    scalars.push_back(scalar);
}

scalars_desc make_dense_push_constants(const std::array<uint32_t, metadata_words>& metadata, bool fused) {
    scalars_desc result;
    result.reserve(fused ? fused_dense_push_constant_words : dense_push_constant_words);
    const auto tensor_offset = [&metadata](shader_abi::tensor_index tensor) {
        return metadata[header_words + shader_abi::index(tensor) * tensor_words + max_rank * 2];
    };
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::element_count)]);
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::input0));
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::input1));
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::output));
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::python_division)]);
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::infinity_detection)]);
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::input0_coefficient)]);
    append_u32_scalar(result, metadata[shader_abi::index(shader_abi::metadata_field::input1_coefficient)]);
    if (fused) {
        append_u32_scalar(result, metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_offset)]);
        append_u32_scalar(result, metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::python_division)]);
        append_u32_scalar(result, metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input0_coefficient)]);
        append_u32_scalar(result, metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input1_coefficient)]);
    }
    return result;
}

std::shared_ptr<kernel_string> make_kernel_source(kernel_kind kind) {
    auto source = std::make_shared<kernel_string>();
    const uint32_t* spirv = eltwise_spirv;
    size_t spirv_size = sizeof(eltwise_spirv);
    if (kind == kernel_kind::dense) {
        spirv = eltwise_dense_spirv;
        spirv_size = sizeof(eltwise_dense_spirv);
    } else if (kind == kernel_kind::dense_push_constants) {
        spirv = eltwise_dense_push_constants_spirv;
        spirv_size = sizeof(eltwise_dense_push_constants_spirv);
    } else if (kind == kernel_kind::dense_f32_vec2_push_constants) {
        spirv = eltwise_dense_f32_vec2_push_constants_spirv;
        spirv_size = sizeof(eltwise_dense_f32_vec2_push_constants_spirv);
    } else if (kind == kernel_kind::dense_f32_vec4_push_constants) {
        spirv = eltwise_dense_f32_vec4_push_constants_spirv;
        spirv_size = sizeof(eltwise_dense_f32_vec4_push_constants_spirv);
    } else if (kind == kernel_kind::dense_f32_vec2_no_tail_push_constants) {
        spirv = eltwise_dense_f32_vec2_no_tail_push_constants_spirv;
        spirv_size = sizeof(eltwise_dense_f32_vec2_no_tail_push_constants_spirv);
    } else if (kind == kernel_kind::dense_f32_vec4_no_tail_push_constants) {
        spirv = eltwise_dense_f32_vec4_no_tail_push_constants_spirv;
        spirv_size = sizeof(eltwise_dense_f32_vec4_no_tail_push_constants_spirv);
    } else if (kind == kernel_kind::dense_packed_8bit_push_constants) {
        spirv = eltwise_dense_packed_8bit_push_constants_spirv;
        spirv_size = sizeof(eltwise_dense_packed_8bit_push_constants_spirv);
    } else if (kind == kernel_kind::dense_packed_16bit_push_constants) {
        spirv = eltwise_dense_packed_16bit_push_constants_spirv;
        spirv_size = sizeof(eltwise_dense_packed_16bit_push_constants_spirv);
    } else if (kind == kernel_kind::dense_packed_f16_push_constants) {
        spirv = eltwise_dense_packed_f16_push_constants_spirv;
        spirv_size = sizeof(eltwise_dense_packed_f16_push_constants_spirv);
    } else if (kind == kernel_kind::broadcast_vector) {
        spirv = eltwise_broadcast_vector_spirv;
        spirv_size = sizeof(eltwise_broadcast_vector_spirv);
    } else if (kind == kernel_kind::unary) {
        spirv = eltwise_unary_spirv;
        spirv_size = sizeof(eltwise_unary_spirv);
    } else if (kind == kernel_kind::scalar_constant) {
        spirv = eltwise_scalar_constant_spirv;
        spirv_size = sizeof(eltwise_scalar_constant_spirv);
    } else if (kind == kernel_kind::fused_dense) {
        spirv = eltwise_fused_dense_spirv;
        spirv_size = sizeof(eltwise_fused_dense_spirv);
    } else if (kind == kernel_kind::fused_dense_push_constants) {
        spirv = eltwise_fused_dense_push_constants_spirv;
        spirv_size = sizeof(eltwise_fused_dense_push_constants_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec2_push_constants) {
        spirv = eltwise_fused_dense_f32_vec2_push_constants_spirv;
        spirv_size = sizeof(eltwise_fused_dense_f32_vec2_push_constants_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec4_push_constants) {
        spirv = eltwise_fused_dense_f32_vec4_push_constants_spirv;
        spirv_size = sizeof(eltwise_fused_dense_f32_vec4_push_constants_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec2_no_tail_push_constants) {
        spirv = eltwise_fused_dense_f32_vec2_no_tail_push_constants_spirv;
        spirv_size = sizeof(eltwise_fused_dense_f32_vec2_no_tail_push_constants_spirv);
    } else if (kind == kernel_kind::fused_dense_f32_vec4_no_tail_push_constants) {
        spirv = eltwise_fused_dense_f32_vec4_no_tail_push_constants_spirv;
        spirv_size = sizeof(eltwise_fused_dense_f32_vec4_no_tail_push_constants_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_8bit_push_constants) {
        spirv = eltwise_fused_dense_packed_8bit_push_constants_spirv;
        spirv_size = sizeof(eltwise_fused_dense_packed_8bit_push_constants_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_16bit_push_constants) {
        spirv = eltwise_fused_dense_packed_16bit_push_constants_spirv;
        spirv_size = sizeof(eltwise_fused_dense_packed_16bit_push_constants_spirv);
    } else if (kind == kernel_kind::fused_dense_packed_f16_push_constants) {
        spirv = eltwise_fused_dense_packed_f16_push_constants_spirv;
        spirv_size = sizeof(eltwise_fused_dense_packed_f16_push_constants_spirv);
    }
    source->str.assign(reinterpret_cast<const char*>(spirv), spirv_size);
    source->entry_point = "main";
    source->batch_compilation = false;
    source->language = kernel_language::SPIRV;
    return source;
}

struct eltwise_impl : typed_primitive_impl<eltwise> {
    using parent = typed_primitive_impl<eltwise>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::eltwise_impl)

    eltwise_impl() : eltwise_impl(kernel_kind::broadcast_scalar, std::nullopt) {}

    eltwise_impl(kernel_kind kind, std::optional<scalar_constant> scalar)
        : parent("vulkan_eltwise"),
          _kernel_source(make_kernel_source(kind)),
          _kernel_kind(kind),
          _scalar_constant(std::move(scalar)) {}

    std::unique_ptr<primitive_impl> clone() const override {
        auto result = std::make_unique<eltwise_impl>(*this);
        result->_metadata_initialized = false;
        return result;
    }

    bool is_cpu() const override {
        return false;
    }

    bool requires_lockable_input() const override {
        return false;
    }

    void save(BinaryOutputBuffer& buffer) const override {
        parent::save(buffer);
        buffer << make_data(&_kernel_kind, sizeof(_kernel_kind));
        const bool has_scalar_constant = _scalar_constant.has_value();
        buffer << make_data(&has_scalar_constant, sizeof(has_scalar_constant));
        if (has_scalar_constant) {
            buffer << make_data(&_scalar_constant->input_index, sizeof(_scalar_constant->input_index));
            buffer << make_data(_scalar_constant->bits.data(), sizeof(_scalar_constant->bits));
        }
    }

    void load(BinaryInputBuffer& buffer) override {
        parent::load(buffer);
        buffer >> make_data(&_kernel_kind, sizeof(_kernel_kind));
        bool has_scalar_constant = false;
        buffer >> make_data(&has_scalar_constant, sizeof(has_scalar_constant));
        if (has_scalar_constant) {
            _scalar_constant.emplace();
            buffer >> make_data(&_scalar_constant->input_index, sizeof(_scalar_constant->input_index));
            buffer >> make_data(_scalar_constant->bits.data(), sizeof(_scalar_constant->bits));
        } else {
            _scalar_constant.reset();
        }
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        if (uses_dense_push_constants(_kernel_kind)) {
            return {};
        }
        return {BufferDescriptor(metadata_words, ov::element::u32, true, false)};
    }

    void init_kernels(const kernels_cache& cache, const kernel_impl_params& params) override {
        _kernels = cache.get_kernels(params);
        OPENVINO_ASSERT(_kernels.size() == 1, "[GPU][Vulkan] Eltwise expects exactly one selected SPIR-V kernel");
    }

    void init_by_cached_kernels(const kernels_cache& cache, std::vector<std::string>& cached_kernel_ids) override {
        _kernels.clear();
        for (const auto& id : cached_kernel_ids) {
            _kernels.push_back(cache.get_kernel_from_cached_kernels(id));
        }
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& cache) override {
        return cache.get_cached_kernel_ids(_kernels);
    }

    std::vector<std::shared_ptr<kernel_string>> get_kernels_source() override {
        return _kernel_source == nullptr ? std::vector<std::shared_ptr<kernel_string>>{} : std::vector<std::shared_ptr<kernel_string>>{_kernel_source};
    }

    void reset_kernels_source() override {
        _kernel_source.reset();
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    void set_kernels(kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "[GPU][Vulkan] Eltwise expects one compiled kernel set");
        const auto& entries = kernels.begin()->second;
        OPENVINO_ASSERT(entries.size() == 1, "[GPU][Vulkan] Eltwise expects exactly one selected compiled kernel");
        _kernels.resize(entries.size());
        for (const auto& entry : entries) {
            _kernels.at(entry.second) = entry.first;
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, eltwise_inst& instance) override {
        OPENVINO_ASSERT(_kernels.size() == 1 && _kernels.front() != nullptr, "[GPU][Vulkan] Eltwise kernel was not initialized");
        const auto desc = instance.get_typed_desc<eltwise>();
        const auto base_input_count = is_unary_mode(desc->mode) ? 1U : 2U;
        const auto fused = get_supported_fused_eltwise(instance.get_node());
        const bool use_fused_dense_kernel = is_fused_dense_kernel(_kernel_kind);
        const bool use_dense_push_constants = uses_dense_push_constants(_kernel_kind);
        OPENVINO_ASSERT(use_fused_dense_kernel == fused.has_value(), "[GPU][Vulkan] Fused Eltwise kernel and graph metadata are inconsistent");
        OPENVINO_ASSERT(instance.inputs_memory_count() == base_input_count && instance.get_fused_mem_count() == (use_fused_dense_kernel ? 1U : 0U),
                        "[GPU][Vulkan] Eltwise received an unexpected regular or fused input count");
        OPENVINO_ASSERT(instance.get_intermediates_memories().size() == (use_dense_push_constants ? 0U : 1U),
                        "[GPU][Vulkan] Eltwise metadata resources do not match the selected kernel ABI");

        auto& stream = instance.get_network().get_stream();
        const auto metadata = make_metadata(instance, _scalar_constant, fused);
        memory::ptr metadata_memory;
        if (!use_dense_push_constants) {
            metadata_memory = instance.get_intermediates_memories().front();
            if (!_metadata_initialized || metadata != _cached_metadata) {
                metadata_memory->copy_from(stream, metadata.data(), true);
                _cached_metadata = metadata;
                _metadata_initialized = true;
            }
        }

        kernel_arguments_desc descriptor;
        descriptor.layerID = instance.id();
        const auto element_count = metadata[shader_abi::index(shader_abi::metadata_field::element_count)];
        const auto& input0_layout = instance.get_input_layout(0);
        const auto& input1_layout = base_input_count == 1 ? input0_layout : instance.get_input_layout(1);
        const auto& output_layout = instance.get_output_layout(0);
        const bool use_dense_kernel = is_plain_dense_kernel(_kernel_kind) || use_fused_dense_kernel;
        const auto* fused_input_layout = use_fused_dense_kernel ? &instance.get_input_layout(fused->external_dependency_index) : nullptr;
        const auto selected_dense_vector_width = get_dense_vector_width(_kernel_kind);
        const auto selected_packed_dense_width = get_packed_dense_width(_kernel_kind);
        const bool use_broadcast_vector_kernel = _kernel_kind == kernel_kind::broadcast_vector;
        const bool use_unary_kernel = _kernel_kind == kernel_kind::unary;
        const bool use_scalar_constant_kernel = _kernel_kind == kernel_kind::scalar_constant;
        OPENVINO_ASSERT(use_scalar_constant_kernel == _scalar_constant.has_value(),
                        "[GPU][Vulkan] Scalar Eltwise kernel and constant metadata are inconsistent");
        OPENVINO_ASSERT(!is_plain_dense_kernel(_kernel_kind) ||
                            (is_packed_dense_kernel(_kernel_kind) ? can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, nullptr)
                                                                  : can_use_dense_kernel(input0_layout, input1_layout, output_layout)),
                        "[GPU][Vulkan] Dense Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(!use_fused_dense_kernel || (is_packed_dense_kernel(_kernel_kind)
                                                        ? can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, fused_input_layout)
                                                        : can_use_fused_dense_kernel(input0_layout, input1_layout, *fused_input_layout, output_layout)),
                        "[GPU][Vulkan] Fused dense Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(selected_dense_vector_width == dense_vector_width::scalar ||
                            can_use_f32_dense_vector_width(input0_layout, input1_layout, output_layout, fused_input_layout, selected_dense_vector_width),
                        "[GPU][Vulkan] F32 vector Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(!use_broadcast_vector_kernel || can_use_broadcast_vector_kernel(output_layout),
                        "[GPU][Vulkan] Vector broadcast Eltwise runtime layout no longer satisfies the compiled kernel contract");
        const auto elements_per_invocation = selected_packed_dense_width != packed_dense_width::none     ? value(selected_packed_dense_width)
                                             : selected_dense_vector_width != dense_vector_width::scalar ? value(selected_dense_vector_width)
                                             : use_fused_dense_kernel                                    ? fused_dense_elements_per_invocation(output_layout)
                                             : use_dense_kernel                                          ? dense_elements_per_invocation(output_layout)
                                             : use_broadcast_vector_kernel                               ? broadcast_vector_width
                                             : use_scalar_constant_kernel ? scalar_constant_elements_per_invocation(output_layout)
                                                                          : output_elements_per_invocation(output_layout);
        const auto invocation_count = (element_count + elements_per_invocation - 1) / elements_per_invocation;
        descriptor.workGroups.global = {invocation_count, 1, 1};
        descriptor.workGroups.local = {
            select_local_work_group_size(invocation_count, instance.get_network().get_engine().get_device_info().max_work_group_size),
            1,
            1};
        descriptor.specialize_local_size_x = true;
        descriptor.specialization_constants = {
            {shader_abi::index(shader_abi::specialization_id::mode), metadata[shader_abi::index(shader_abi::metadata_field::mode)]},
            {shader_abi::index(shader_abi::specialization_id::input0_type), metadata[shader_abi::index(shader_abi::metadata_field::input0_type)]},
            {shader_abi::index(shader_abi::specialization_id::input1_type), metadata[shader_abi::index(shader_abi::metadata_field::input1_type)]},
            {shader_abi::index(shader_abi::specialization_id::output_type), metadata[shader_abi::index(shader_abi::metadata_field::output_type)]},
            {shader_abi::index(shader_abi::specialization_id::storage_flags), metadata[shader_abi::index(shader_abi::metadata_field::flags)]},
        };
        if (use_dense_kernel || use_broadcast_vector_kernel || use_scalar_constant_kernel) {
            descriptor.specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::elements_per_invocation), elements_per_invocation});
        }
        if (use_fused_dense_kernel) {
            descriptor.specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_mode),
                                                           metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::mode)]});
            descriptor.specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_input_type),
                                                           metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_type)]});
            descriptor.specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::fused_input_position),
                 metadata[fused_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_position)]});
        }
        if (use_dense_push_constants) {
            descriptor.scalars = make_dense_push_constants(metadata, use_fused_dense_kernel);
        }
        descriptor.arguments = {{argument_desc::Types::INPUT, 0}};
        if (use_scalar_constant_kernel) {
            descriptor.arguments.front().index = _scalar_constant->input_index == shader_abi::tensor_index::input0
                                                     ? shader_abi::index(shader_abi::tensor_index::input1)
                                                     : shader_abi::index(shader_abi::tensor_index::input0);
        } else if (!use_unary_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::INPUT, 1});
        }
        if (use_fused_dense_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE, 0});
        }
        descriptor.arguments.push_back({argument_desc::Types::OUTPUT, 0});
        if (!use_dense_push_constants) {
            descriptor.arguments.push_back({argument_desc::Types::INTERNAL_BUFFER, 0});
        }
        if (!use_dense_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::OUTPUT, 0});
        }

        kernel_arguments_data arguments;
        for (size_t input_index = 0; input_index < base_input_count; ++input_index) {
            arguments.inputs.push_back(instance.input_memory_ptr(input_index));
        }
        if (use_fused_dense_kernel) {
            arguments.fused_op_inputs.push_back(instance.fused_memory(0));
        }
        arguments.outputs = {instance.output_memory_ptr(0)};
        if (!use_dense_push_constants) {
            arguments.intermediates = {metadata_memory};
        }
        return stream.enqueue_kernel(*_kernels.front(), descriptor, arguments, events, instance.needs_completion_event());
    }

private:
    std::shared_ptr<kernel_string> _kernel_source;
    kernel_kind _kernel_kind = kernel_kind::broadcast_scalar;
    std::optional<scalar_constant> _scalar_constant;
    std::vector<kernel::ptr> _kernels;
    std::array<uint32_t, metadata_words> _cached_metadata{};
    bool _metadata_initialized = false;
};

}  // namespace

bool EltwiseImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    if (node.get_program().get_engine().runtime_type() != runtime_types::vulkan) {
        return false;
    }

    const auto& desc = node.as<eltwise>().get_primitive();
    const auto expected_inputs = is_unary_mode(desc->mode) ? 1U : 2U;
    const auto fused = get_supported_fused_eltwise(node);
    if (node.has_fused_primitives() && !fused.has_value()) {
        return false;
    }
    const bool coefficients_supported = desc->coefficients.empty() || (desc->mode == eltwise_mode::sum && desc->coefficients.size() == expected_inputs) ||
                                        (desc->mode == eltwise_mode::is_inf && desc->coefficients.size() == 2);
    if (!is_supported_mode(desc->mode) || node.get_dependencies().size() != expected_inputs + (fused.has_value() ? 1U : 0U) || !coefficients_supported ||
        !desc->stride.empty() ||
        (desc->broadcast_spec.m_type != ov::op::AutoBroadcastType::NUMPY && desc->broadcast_spec.m_type != ov::op::AutoBroadcastType::NONE)) {
        return false;
    }

    for (size_t index = 0; index < expected_inputs; ++index) {
        const auto& input_layout = node.get_input_layout(index);
        if (!is_supported_data_type(input_layout.data_type) || !is_supported_format(input_layout.format.value) || input_layout.get_rank() > max_rank) {
            return false;
        }
        if (is_bitwise_mode(desc->mode) && !is_integer_data_type(input_layout.data_type)) {
            return false;
        }
        if ((desc->mode == eltwise_mode::atan2 || is_unary_mode(desc->mode)) && !data_type_traits::is_floating_point(input_layout.data_type)) {
            return false;
        }
    }
    const auto& output_layout = node.get_output_layout(0);
    return is_supported_data_type(output_layout.data_type) && is_supported_format(output_layout.format.value) && output_layout.get_rank() <= max_rank;
}

std::unique_ptr<primitive_impl> EltwiseImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    const auto* engine = dynamic_cast<const vulkan_engine*>(&node.get_program().get_engine());
    OPENVINO_ASSERT(engine != nullptr, "[GPU][Vulkan] Eltwise implementation requires a Vulkan engine");
    const auto max_push_constants_size = engine->get_max_push_constants_size();
    const auto& device_info = engine->get_device_info();
    const auto input_count = params.input_layouts.size();
    const auto fused = get_supported_fused_eltwise(node);
    OPENVINO_ASSERT(input_count == 1 || input_count == 2 || (input_count == 3 && fused.has_value()),
                    "[GPU][Vulkan] Eltwise implementation creation received an unexpected input count");
    const auto& input0_layout = params.get_input_layout(0);
    const auto& input1_layout = input_count == 1 ? input0_layout : params.get_input_layout(1);
    kernel_kind kind = kernel_kind::broadcast_scalar;
    std::optional<scalar_constant> scalar;
    if (fused.has_value()) {
        kind = kernel_kind::fused_dense;
        if (max_push_constants_size >= fused_dense_push_constant_bytes) {
            const auto& fused_input_layout = params.get_input_layout(fused->external_dependency_index);
            const auto packed_width = select_packed_dense_width(input0_layout, input1_layout, params.get_output_layout(0), &fused_input_layout, device_info);
            if (packed_width != packed_dense_width::none) {
                kind = select_packed_dense_kernel_kind(packed_width, params.get_output_layout(0).data_type, true);
            } else {
                const auto vector_width =
                    select_f32_dense_vector_width(input0_layout, input1_layout, params.get_output_layout(0), &fused_input_layout, device_info);
                const auto no_tail =
                    vector_width != dense_vector_width::scalar && can_use_f32_no_tail_kernel(params.get_output_layout(0), vector_width, device_info);
                kind = vector_width == dense_vector_width::scalar ? kernel_kind::fused_dense_push_constants
                                                                  : select_f32_vector_kernel_kind(vector_width, true, no_tail);
            }
        }
    } else if (is_unary_mode(node.as<eltwise>().get_primitive()->mode)) {
        kind = kernel_kind::unary;
    } else if (!params.is_dynamic()) {
        const auto& output_layout = params.get_output_layout(0);
        scalar = get_scalar_constant(node);
        if (scalar.has_value() && output_layout.count() >= broadcast_vector_min_elements) {
            kind = kernel_kind::scalar_constant;
        } else if (can_use_dense_kernel(input0_layout, input1_layout, output_layout) ||
                   can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, nullptr)) {
            kind = can_use_dense_kernel(input0_layout, input1_layout, output_layout) ? kernel_kind::dense : kernel_kind::broadcast_scalar;
            if (max_push_constants_size >= dense_push_constant_bytes) {
                const auto packed_width = select_packed_dense_width(input0_layout, input1_layout, output_layout, nullptr, device_info);
                if (packed_width != packed_dense_width::none) {
                    kind = select_packed_dense_kernel_kind(packed_width, output_layout.data_type, false);
                } else if (can_use_dense_kernel(input0_layout, input1_layout, output_layout)) {
                    const auto vector_width = select_f32_dense_vector_width(input0_layout, input1_layout, output_layout, nullptr, device_info);
                    const auto no_tail = vector_width != dense_vector_width::scalar && can_use_f32_no_tail_kernel(output_layout, vector_width, device_info);
                    kind = vector_width == dense_vector_width::scalar ? kernel_kind::dense_push_constants
                                                                      : select_f32_vector_kernel_kind(vector_width, false, no_tail);
                } else {
                    kind = kernel_kind::broadcast_scalar;
                }
            }
        } else if (can_use_broadcast_vector_kernel(output_layout)) {
            kind = kernel_kind::broadcast_vector;
        }
        if (kind != kernel_kind::scalar_constant) {
            scalar.reset();
        }
    }
    return std::make_unique<eltwise_impl>(kind, std::move(scalar));
}

}  // namespace vulkan
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::eltwise_impl)
