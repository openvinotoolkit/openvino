// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_ARM64)

#include <memory>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>

#include "nodes/topk.h"

namespace ov::intel_cpu::node {

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
struct jit_uni_topk_kernel_aarch64;

std::shared_ptr<jit_uni_topk_kernel> create_topk_kernel_aarch64(const jit_topk_config_params& jcp);

}  // namespace ov::intel_cpu::node

#endif  // defined(OPENVINO_ARCH_ARM64)
