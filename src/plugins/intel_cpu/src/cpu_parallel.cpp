// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_parallel.hpp"

#include <cstddef>
#include <memory>

#include "openvino/runtime/intel_cpu/properties.hpp"
#include "thread_pool_imp.hpp"

namespace ov::intel_cpu {
CpuParallel::CpuParallel(ov::intel_cpu::TbbPartitioner partitioner, size_t multiplier)
    : m_partitioner(partitioner),
      m_multiplier(multiplier) {
    m_partitioner =
        m_partitioner == ov::intel_cpu::TbbPartitioner::NONE ? ov::intel_cpu::TbbPartitioner::STATIC : m_partitioner;
    m_thread_pool = std::make_shared<ThreadPool>(*this);
}

}  // namespace ov::intel_cpu
