// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <vector>

#include "common_utils/kernels_cache.hpp"
#include "openvino/core/except.hpp"

namespace cldnn {

/// Owns compiled kernel handles and the compilation/cache lifecycle shared by GPU implementations.
///
/// Execution code accesses precomputed kernel handles by index; this helper performs no work in the dispatch path.
class gpu_kernel_lifecycle final {
public:
    using container_type = std::vector<kernel::ptr>;

    void clone_from(const gpu_kernel_lifecycle& other, bool share_kernel_handles) {
        _kernels.clear();
        _kernels.reserve(other.size());
        for (const auto& kernel : other._kernels) {
            _kernels.emplace_back(kernel->clone(share_kernel_handles));
        }
    }

    bool initialize(const kernels_cache& cache, const kernel_impl_params& params) {
        _kernels = cache.get_kernels(params);
        return cache.get_kernels_reuse();
    }

    bool restore(const kernels_cache& cache, const std::vector<std::string>& cached_kernel_ids) {
        _kernels.clear();
        _kernels.reserve(cached_kernel_ids.size());
        for (const auto& id : cached_kernel_ids) {
            _kernels.emplace_back(cache.get_kernel_from_cached_kernels(id));
        }
        return cache.get_kernels_reuse();
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& cache) const {
        return cache.get_cached_kernel_ids(_kernels);
    }

    void adopt_compiled(kernels_cache::compiled_kernels compiled) {
        OPENVINO_ASSERT(compiled.size() == 1, "[GPU] Expected one compiled kernel set");
        adopt_entries(compiled.begin()->second);
    }

    void adopt_entries(const std::vector<std::pair<kernel::ptr, size_t>>& entries) {
        _kernels.clear();
        _kernels.resize(entries.size());
        for (const auto& entry : entries) {
            OPENVINO_ASSERT(entry.second < _kernels.size(), "[GPU] Compiled kernel index is out of range");
            _kernels[entry.second] = entry.first;
        }
    }

    container_type copy_kernels() const {
        return _kernels;
    }

    size_t size() const noexcept {
        return _kernels.size();
    }
    void clear() noexcept {
        _kernels.clear();
    }
    const kernel::ptr& operator[](size_t index) const {
        return _kernels[index];
    }
    const kernel::ptr& at(size_t index) const {
        return _kernels.at(index);
    }
    const kernel::ptr& front() const {
        return _kernels.front();
    }

    template <typename... Args>
    kernel::ptr& emplace_back(Args&&... args) {
        return _kernels.emplace_back(std::forward<Args>(args)...);
    }

private:
    container_type _kernels;
};

}  // namespace cldnn
