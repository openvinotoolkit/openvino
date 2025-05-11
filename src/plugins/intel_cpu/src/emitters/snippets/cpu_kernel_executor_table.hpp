// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "snippets/kernel_executor_table.hpp"

namespace ov::intel_cpu {

template <typename Conf, typename KernelType>
class CPUKernelExecutor : public snippets::KernelExecutor<Conf, KernelType> {
public:
    CPUKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, Conf c)
        : snippets::KernelExecutor<Conf, KernelType>(std::move(c)),
          m_kernel_cache(std::move(kernel_cache)) {}

    void update_kernel(const Conf& config, std::shared_ptr<KernelType>& kernel) const override final {
        const auto& cache = m_kernel_cache.lock();
        OPENVINO_ASSERT(cache, "Invalid kernel cache pointer in CPUKernelExecutor::update_kernel()");
        const auto& lookup_result = cache->getOrCreate(Key(config), [this](const Key& k) {
            return compile_kernel(k.config);
        });
        kernel = lookup_result.first;
    }

protected:
    struct Key {
        explicit Key(Conf c) : config{std::move(c)} {}
        const Conf config;
        [[nodiscard]] size_t hash() const {
            return config.hash();
        }
        bool operator==(const Key& rhs) const {
            return config == rhs.config;
        }
    };
    /** Compile kernel managed by KernelExecutor instance. Will be called only if Kernel is not found in the cache */
    virtual std::shared_ptr<KernelType> compile_kernel(const Conf& c) const = 0;
    /** CPU plugin cache implementation is used to avoid redundant recompilations */
    ov::intel_cpu::MultiCacheWeakPtr m_kernel_cache;
};

}  // namespace ov::intel_cpu
