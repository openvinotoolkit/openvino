// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "cpu_parallel.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

void reorder_kv_cache(ov::intel_cpu::PlainTensor& key_cache,
                      ov::intel_cpu::PlainTensor& value_cache,
                      const ov::intel_cpu::PlainTensor& block_indices,
                      const ov::intel_cpu::PlainTensor& block_indices_begins,
                      const ov::intel_cpu::PlainTensor& block_update_indices,
                      const ov::intel_cpu::PlainTensor& block_update_indices_begins,
                      bool key_by_channel,
                      bool value_by_channel,
                      const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
