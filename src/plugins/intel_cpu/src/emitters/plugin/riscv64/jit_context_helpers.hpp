// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64::utils {

size_t get_vlen_bytes();

void sub_sp(jit_generator_t& h, size_t bytes);
void add_sp(jit_generator_t& h, size_t bytes);

void save_vector_state(jit_generator_t& h,
                       const Xbyak_riscv::Reg& vl_gpr,
                       const Xbyak_riscv::Reg& vtype_gpr,
                       size_t vl_offset,
                       size_t vtype_offset);
void restore_vector_state(jit_generator_t& h,
                          const Xbyak_riscv::Reg& vl_gpr,
                          const Xbyak_riscv::Reg& vtype_gpr,
                          size_t vl_offset,
                          size_t vtype_offset);

void save_vregs(jit_generator_t& h,
                const Xbyak_riscv::Reg& vlen_gpr,
                const Xbyak_riscv::Reg& ptr_gpr,
                size_t stack_offset,
                const std::vector<size_t>& vreg_idxs);
void restore_vregs(jit_generator_t& h,
                   const Xbyak_riscv::Reg& vlen_gpr,
                   const Xbyak_riscv::Reg& ptr_gpr,
                   size_t stack_offset,
                   const std::vector<size_t>& vreg_idxs);

}  // namespace ov::intel_cpu::riscv64::utils
