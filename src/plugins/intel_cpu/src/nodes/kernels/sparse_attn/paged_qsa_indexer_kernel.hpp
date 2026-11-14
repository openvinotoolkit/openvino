// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "nodes/kernels/sparse_attn/qsa_executor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// The PagedQSAIndexer executor of `config`, cross-compiled: AVX-512, AVX2 or plain C++ by the CPU.
std::shared_ptr<ov::Extensions::Cpu::qsa::PagedExecutor> make_paged_qsa_executor(
    const ov::Extensions::Cpu::qsa::Config& config);

}  // namespace ov::Extensions::Cpu::XARCH
