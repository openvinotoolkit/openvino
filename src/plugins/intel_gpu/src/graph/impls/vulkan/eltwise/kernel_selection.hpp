// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "../eltwise_shader_abi.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "kernel_kind.hpp"

namespace cldnn {

struct program_node;

namespace vulkan::eltwise_detail {

struct scalar_constant {
    eltwise_shader_abi::tensor_index input_index = eltwise_shader_abi::tensor_index::input1;
    std::array<uint32_t, 2> bits{};
};

bool is_supported_format(format::type fmt);
uint32_t checked_u32(size_t value, const char* description);
bool has_dense_storage(const layout& tensor_layout);
std::optional<scalar_constant> get_scalar_constant(const program_node& node);
bool can_use_scalar_linear_storage(const layout& tensor_layout, const layout& output_layout);
bool can_use_linear_storage(const layout& input0_layout, const layout& input1_layout, const layout& output_layout);
uint32_t output_elements_per_invocation(const layout& output_layout);
bool can_use_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& output_layout);
bool can_use_packed_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& output_layout, const layout* fused_input_layout);
packed_dense_width select_packed_dense_width(const layout& input0_layout,
                                             const layout& input1_layout,
                                             const layout& output_layout,
                                             const layout* fused_input_layout,
                                             const device_info& info);
kernel_kind select_packed_dense_kernel_kind(packed_dense_width width, data_types type, bool fused);
bool can_use_fused_dense_kernel(const layout& input0_layout, const layout& input1_layout, const layout& fused_input_layout, const layout& output_layout);
bool can_use_f32_dense_vector_width(const layout& input0_layout,
                                    const layout& input1_layout,
                                    const layout& output_layout,
                                    const layout* fused_input_layout,
                                    dense_vector_width width);
dense_vector_width select_f32_dense_vector_width(const layout& input0_layout,
                                                 const layout& input1_layout,
                                                 const layout& output_layout,
                                                 const layout* fused_input_layout,
                                                 const device_info& info);
bool can_use_f32_no_tail_kernel(const layout& output_layout, dense_vector_width width, const device_info& info);
kernel_kind select_f32_vector_kernel_kind(dense_vector_width width, bool fused, bool no_tail);
kernel_kind select_pre_specialized_kernel_kind(kernel_kind fallback,
                                               eltwise_mode mode,
                                               const layout& input0_layout,
                                               const layout& input1_layout,
                                               const layout& output_layout);
uint32_t select_generic_elements_per_invocation(kernel_kind kind,
                                                const layout& input0_layout,
                                                const layout& input1_layout,
                                                const layout& output_layout,
                                                const layout* fused_input_layout,
                                                eltwise_mode mode,
                                                const std::vector<eltwise_mode>& fused_modes,
                                                const device_info& info);
bool can_use_broadcast_vector_kernel(const layout& output_layout);
bool benefits_from_scalar_constant_kernel(const layout& output_layout);
bool should_use_fast_broadcast_kernel(const layout& input0_layout,
                                      const layout& input1_layout,
                                      const layout& output_layout,
                                      const device_info& info,
                                      uint32_t elements_per_invocation);

}  // namespace vulkan::eltwise_detail
}  // namespace cldnn
