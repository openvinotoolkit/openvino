// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64::jit_conversion {

bool is_supported_convert_precision(const ov::element::Type& precision);
void validate_convert_precision(const ov::element::Type& input_type, const ov::element::Type& output_type);

void emit_convert_process(ov::intel_cpu::riscv64::jit_generator_t* h,
                          const Xbyak_riscv::VReg& src,
                          const Xbyak_riscv::VReg& dst,
                          const ov::element::Type& input_type,
                          const ov::element::Type& output_type,
                          bool saturation,
                          const std::vector<size_t>& aux_gpr_idxs);

}  // namespace ov::intel_cpu::riscv64::jit_conversion
