// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <nodes/kernels/riscv64/jit_generator.hpp>

#include "openvino/core/type/element_type.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64 {

enum class arithmetic_mode : uint8_t { saturation, truncation };

namespace jit_conversion {

bool is_supported_convert_precision(const ov::element::Type& precision);
void validate_convert_precision(const ov::element::Type& input_type, const ov::element::Type& output_type);
Xbyak_riscv::SEW byte_size_to_sew(size_t byte_size);

void emit_convert_process(ov::intel_cpu::riscv64::jit_generator_t* h,
                          const Xbyak_riscv::VReg& src,
                          const Xbyak_riscv::VReg& dst,
                          const ov::element::Type& input_type,
                          const ov::element::Type& output_type,
                          arithmetic_mode mode,
                          const Xbyak_riscv::Reg& avl);

}  // namespace jit_conversion
}  // namespace ov::intel_cpu::riscv64
