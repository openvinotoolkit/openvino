// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selection.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "data_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "local_size_tuner.hpp"
#include "metadata_builder.hpp"
#include "openvino/core/except.hpp"

namespace cldnn::vulkan::eltwise_detail {
namespace {

constexpr uint32_t broadcast_vector_width = 4;
constexpr uint32_t broadcast_vector_min_elements = 256;
constexpr uint32_t maximum_scalar_batch_width = 8;
constexpr uint32_t balanced_scalar_batch_width = 4;
constexpr uint32_t division_scalar_batch_width = 2;
constexpr uint32_t scalar_batch_width = 1;
constexpr uint32_t scalar_elements_per_subgroup_budget = portable_local_work_group_size_limit;

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

bool is_aligned_for_vector_width(const layout& tensor_layout, dense_vector_width width) {
    return tensor_layout.get_linear_offset() % value(width) == 0;
}

uint32_t operation_batch_width_limit(eltwise_mode mode, data_types type, kernel_kind kind, const device_info& info) {
    switch (mode) {
    case eltwise_mode::pow:
    case eltwise_mode::atan2:
        return scalar_batch_width;
    case eltwise_mode::div:
    case eltwise_mode::mod:
    case eltwise_mode::floor_mod:
        if (type == data_types::i64) {
            return scalar_batch_width;
        }
        if (is_broadcast_vector_kernel(kind) && info.max_work_group_size != 0) {
            const auto capability_width = std::max<uint64_t>(info.max_work_group_size / portable_local_work_group_size_limit, scalar_batch_width);
            return static_cast<uint32_t>(std::min<uint64_t>(capability_width, balanced_scalar_batch_width));
        }
        return division_scalar_batch_width;
    case eltwise_mode::squared_diff:
        return balanced_scalar_batch_width;
    default:
        return maximum_scalar_batch_width;
    }
}

bool has_aligned_offset(const layout& tensor_layout, uint32_t width) {
    return tensor_layout.get_linear_offset() % width == 0;
}

}  // namespace

bool is_supported_format(format::type fmt) {
    return one_of(fmt, {format::any, format::bfyx, format::yxfb, format::bfzyx, format::bfwzyx, format::bfuwzyx, format::bfvuwzyx});
}

uint32_t checked_u32(size_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] Eltwise ", description, " exceeds the 32-bit shader metadata range");
    return static_cast<uint32_t>(value);
}

bool has_dense_storage(const layout& tensor_layout) {
    return !static_cast<bool>(tensor_layout.data_padding) &&
           tensor_layout.bytes_count() == tensor_layout.count() * data_type_traits::size_of(tensor_layout.data_type);
}

std::optional<scalar_constant> get_scalar_constant(const program_node& node) {
    for (uint32_t input_index = eltwise_shader_abi::index(eltwise_shader_abi::tensor_index::input0);
         input_index <= eltwise_shader_abi::index(eltwise_shader_abi::tensor_index::input1);
         ++input_index) {
        const auto& dependency = node.get_dependency(input_index);
        const auto& input_layout = dependency.get_output_layout();
        if (!dependency.is_type<data>() || !dependency.is_constant() || input_layout.is_dynamic() || input_layout.count() != 1 ||
            input_layout.get_linear_offset() != 0 || !has_dense_storage(input_layout)) {
            continue;
        }

        scalar_constant result;
        result.input_index = static_cast<eltwise_shader_abi::tensor_index>(input_index);
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

bool can_use_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& output_layout) {
    return data_type_traits::size_of(output_layout.data_type) >= sizeof(uint32_t) && can_use_linear_storage(input0_layout, input1_layout, output_layout);
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
    if (logical_elements_per_subgroup > portable_local_work_group_size_limit) {
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
    const auto local_size = select_portable_local_work_group_size(invocation_count, max_work_group_size);
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

kernel_kind select_pre_specialized_kernel_kind(kernel_kind fallback,
                                               eltwise_mode mode,
                                               const layout& input0_layout,
                                               const layout& input1_layout,
                                               const layout& output_layout) {
    if (fallback == kernel_kind::dense_push_constants && input0_layout.data_type == input1_layout.data_type &&
        input0_layout.data_type == output_layout.data_type) {
        if (input0_layout.data_type == data_types::f32) {
            if (mode == eltwise_mode::sum) {
                return kernel_kind::dense_f32_sum_push_constants;
            }
            if (mode == eltwise_mode::div) {
                return kernel_kind::dense_f32_div_push_constants;
            }
        }
        if (input0_layout.data_type == data_types::i64) {
            if (mode == eltwise_mode::sum) {
                return kernel_kind::dense_i64_sum_push_constants;
            }
            if (mode == eltwise_mode::div) {
                return kernel_kind::dense_i64_div_push_constants;
            }
        }
    }
    if (mode == eltwise_mode::eq && input0_layout.data_type == data_types::f32 && input1_layout.data_type == data_types::f32 &&
        output_layout.data_type == data_types::boolean) {
        if (fallback == kernel_kind::broadcast_scalar) {
            return kernel_kind::broadcast_f32_eq;
        }
        if (fallback == kernel_kind::broadcast_fast_scalar) {
            return kernel_kind::broadcast_fast_f32_eq;
        }
    }
    return fallback;
}

uint32_t select_generic_elements_per_invocation(kernel_kind kind,
                                                const layout& input0_layout,
                                                const layout& input1_layout,
                                                const layout& output_layout,
                                                const layout* fused_input_layout,
                                                eltwise_mode mode,
                                                const std::vector<eltwise_mode>& fused_modes,
                                                const device_info& info) {
    const auto packed_width = get_packed_dense_width(kind);
    if (packed_width != packed_dense_width::none) {
        return value(packed_width);
    }
    const auto vector_width = get_dense_vector_width(kind);
    if (vector_width != dense_vector_width::scalar) {
        return value(vector_width);
    }
    const auto output_width = output_elements_per_invocation(output_layout);
    if (output_width != 1 || kind == kernel_kind::unary || kind == kernel_kind::broadcast_scalar || kind == kernel_kind::broadcast_fast_scalar) {
        return output_width;
    }
    OPENVINO_ASSERT(is_plain_dense_kernel(kind) || is_fused_kernel(kind) || is_fused_post_op_kernel(kind) || is_broadcast_vector_kernel(kind) ||
                        kind == kernel_kind::scalar_constant,
                    "[GPU][Vulkan] Eltwise scalar batch selection received an unsupported kernel kind");

    uint32_t width_limit = std::min({operation_batch_width_limit(mode, input0_layout.data_type, kind, info),
                                     operation_batch_width_limit(mode, input1_layout.data_type, kind, info),
                                     operation_batch_width_limit(mode, output_layout.data_type, kind, info)});
    for (const auto fused_mode : fused_modes) {
        width_limit = std::min(width_limit, operation_batch_width_limit(fused_mode, output_layout.data_type, kind, info));
    }
    if (is_plain_dense_kernel(kind) || is_fused_kernel(kind) || is_fused_post_op_kernel(kind) || kind == kernel_kind::scalar_constant) {
        width_limit = std::min(width_limit, balanced_scalar_batch_width);
    }
    auto maximum_scalar_size = std::max({data_type_traits::size_of(input0_layout.data_type),
                                         data_type_traits::size_of(input1_layout.data_type),
                                         data_type_traits::size_of(output_layout.data_type)});
    if (fused_input_layout != nullptr) {
        maximum_scalar_size = std::max(maximum_scalar_size, data_type_traits::size_of(fused_input_layout->data_type));
    }
    if (maximum_scalar_size >= sizeof(uint64_t) && width_limit >= division_scalar_batch_width) {
        return division_scalar_batch_width;
    }
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0 || info.max_work_group_size == 0) {
        return scalar_batch_width;
    }

    const auto subgroup_size = static_cast<uint64_t>(info.supported_simd_sizes.front());
    const auto subgroup_width_limit = std::max<uint64_t>(scalar_elements_per_subgroup_budget / subgroup_size, balanced_scalar_batch_width);
    width_limit = static_cast<uint32_t>(std::min<uint64_t>(width_limit, subgroup_width_limit));
    const auto max_work_group_size = std::max<uint64_t>(info.max_work_group_size, 1);
    const auto subgroups_per_full_work_group = (max_work_group_size + subgroup_size - 1) / subgroup_size;
    const auto useful_subgroup_coverage = std::min<uint64_t>(subgroups_per_full_work_group, width_limit);
    const auto required_parallel_invocations = max_work_group_size * useful_subgroup_coverage;
    const auto has_aligned_layouts = [&](uint32_t candidate) {
        if (!has_aligned_offset(output_layout, candidate)) {
            return false;
        }
        const bool requires_aligned_inputs = is_plain_dense_kernel(kind) || is_fused_kernel(kind) || is_fused_post_op_kernel(kind);
        return !requires_aligned_inputs ||
               (has_aligned_offset(input0_layout, candidate) && has_aligned_offset(input1_layout, candidate) &&
                (fused_input_layout == nullptr || is_fused_broadcast_kernel(kind) || has_aligned_offset(*fused_input_layout, candidate)));
    };
    constexpr std::array candidates{
        maximum_scalar_batch_width,
        balanced_scalar_batch_width,
        division_scalar_batch_width,
        scalar_batch_width,
    };
    for (const auto candidate : candidates) {
        if (candidate > width_limit || !has_aligned_layouts(candidate)) {
            continue;
        }
        const auto invocation_count = (output_layout.count() + candidate - 1) / candidate;
        if (invocation_count >= required_parallel_invocations) {
            return candidate;
        }
    }
    return scalar_batch_width;
}

bool can_use_broadcast_vector_kernel(const layout& output_layout) {
    if (data_type_traits::size_of(output_layout.data_type) < sizeof(uint32_t) || output_layout.count() < broadcast_vector_min_elements) {
        return false;
    }
    return output_layout.count() % broadcast_vector_width == 0;
}

bool benefits_from_scalar_constant_kernel(const layout& output_layout) {
    return output_layout.count() >= broadcast_vector_min_elements;
}

bool should_use_fast_broadcast_kernel(const layout& input0_layout,
                                      const layout& input1_layout,
                                      const layout& output_layout,
                                      const device_info& info,
                                      uint32_t elements_per_invocation) {
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0 || info.max_work_group_size == 0) {
        return false;
    }
    const auto subgroup_size = info.supported_simd_sizes.front();
    const auto subgroup_slots = std::max<uint64_t>(info.max_work_group_size / subgroup_size, 1);
    const auto rank = EltwiseMetadata::collapsed_broadcast_rank(input0_layout, input1_layout, output_layout);
    const auto coordinate_schedule_pressure = static_cast<uint64_t>(rank) * subgroup_slots;
    const auto invocation_count = (output_layout.count() + elements_per_invocation - 1) / elements_per_invocation;
    return coordinate_schedule_pressure <= portable_local_work_group_size_limit && invocation_count >= subgroup_size;
}

}  // namespace cldnn::vulkan::eltwise_detail
