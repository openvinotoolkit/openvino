// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/jit_generator.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_f16_to_f32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_f32_to_f16(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_f32_to_i32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_i32_to_f32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_i32_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                    bool is_signed, bool is_saturated);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_dbyte_to_i32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                    bool is_signed);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_f16_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_dbyte_to_f16(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                    bool is_signed);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_dbyte_to_byte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                    bool is_signed, bool is_saturated);

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_byte_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                    bool is_signed);

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
