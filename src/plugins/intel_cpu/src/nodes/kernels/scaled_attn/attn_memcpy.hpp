// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void attn_memcpy(const ov::intel_cpu::PlainTensor& k_input,
                const ov::intel_cpu::PlainTensor& v_input,
                const ov::intel_cpu::PlainTensor& past_k_output,
                const ov::intel_cpu::PlainTensor& past_v_output);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov