// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/kernel_executor_table.hpp"
#include "cache/multi_cache.h"

namespace ov {
namespace intel_cpu {

template<typename Conf, typename KernelType>
class CPUKernelExecutor : public snippets::KernelExecutor<Conf, KernelType> {
public:
     CPUKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, std::shared_ptr<Conf> c) :
                       snippets::KernelExecutor<Conf, KernelType>(c), m_kernel_cache(std::move(kernel_cache)) {}
     struct Key {
         explicit Key(std::shared_ptr<Conf> c) : config{std::move(c)} {}
         const std::shared_ptr<Conf> config;
         size_t hash() const { return config->hash(); }
         bool operator==(const Key& rhs) const { return *config == *rhs.config; }
     };
    void update_kernel() override {
        OPENVINO_ASSERT(m_config && m_config->is_completed(), "Update kernel was called with invalid config");
        const auto& cache = m_kernel_cache.lock();
        OPENVINO_ASSERT(cache, "Invalid kernel cache pointer in CPUKernelExecutor::update_kernel()");
        const auto& lookup_result = cache->getOrCreate(Key(m_config),
                                                       [this](const Key& k) {
                                                            return compile_kernel(k.config);
                                                       });
        m_kernel = lookup_result.first;
        OPENVINO_ASSERT(m_kernel, "Failed to compile kernel executor");
    }

protected:
    // Note: this usings are needed because non-dependent names are not looked up in dependent base classes
    using snippets::KernelExecutor<Conf, KernelType>::m_config;
    using snippets::KernelExecutor<Conf, KernelType>::m_kernel;
    using snippets::KernelExecutor<Conf, KernelType>::compile_kernel;
    /** CPU plugin cache implementation is used to avoid redundant recompilations */
    ov::intel_cpu::MultiCacheWeakPtr m_kernel_cache;
};

}   // namespace intel_cpu
}   // namespace ov
