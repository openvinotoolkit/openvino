// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_cpu_blocking.hpp"

#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::pass {

size_t GemmCPUBlocking::get_default_k_blk([[maybe_unused]] size_t k) const {
    return ov::snippets::utils::get_full_dim_value();
}

}  // namespace ov::intel_cpu::pass
