/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_COMPUTE_DEVICE_INFO_HPP
#define GPU_COMPUTE_DEVICE_INFO_HPP

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "common/z_magic.hpp"
#include "cpu/platform.hpp"
#include "oneapi/dnnl/dnnl_config.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

enum class gpu_arch_t {
    unknown,
    gen9,
    xe_lp,
    xe_hp,
    xe_hpg,
};

enum class device_ext_t : uint64_t {
    // clang-format off
    // OpenCL data types
    khr_fp16 = 1ull << 0,
    khr_fp64 = 1ull << 1,
    // OpenCL atomics
    khr_global_int32_base_atomics     = 1ull << 2,
    khr_global_int32_extended_atomics = 1ull << 3,
    khr_int64_base_atomics            = 1ull << 4,
    khr_int64_extended_atomics        = 1ull << 5,
    khr_local_int32_base_atomics      = 1ull << 6,
    khr_local_int32_extended_atomics  = 1ull << 7,
    // Intel specific Gen9+
    intel_subgroups              = 1ull << 16,
    intel_required_subgroup_size = 1ull << 17,
    intel_subgroups_char         = 1ull << 18,
    intel_subgroups_short        = 1ull << 19,
    intel_subgroups_long         = 1ull << 20,
    // Intel specific Xe_LP+
    intel_subgroup_local_block_io = 1ull << 21,
    intel_dot_accumulate          = 1ull << 22,
    // Intel specific Xe_HP+
    intel_global_float_atomics                      = 1ull << 23,
    intel_subgroup_matrix_multiply_accumulate       = 1ull << 24,
    intel_subgroup_split_matrix_multiply_accumulate = 1ull << 25,
    intel_variable_eu_thread_count                  = 1ull << 26,
    // Future extensions
    future_bf16_cvt                                 = 1ull << 31,
    last
    // clang-format on
};

static inline const char *ext2cl_str(device_ext_t ext) {
#define CASE(x) \
    case device_ext_t::x: return STRINGIFY(CONCAT2(cl_, x));
    switch (ext) {
        CASE(khr_fp16)
        CASE(khr_fp64)

        CASE(khr_global_int32_base_atomics)
        CASE(khr_global_int32_extended_atomics)
        CASE(khr_int64_base_atomics)
        CASE(khr_int64_extended_atomics)
        CASE(khr_local_int32_base_atomics)
        CASE(khr_local_int32_extended_atomics)

        CASE(intel_subgroups)
        CASE(intel_required_subgroup_size)
        CASE(intel_subgroups_char)
        CASE(intel_subgroups_short)
        CASE(intel_subgroups_long)

        CASE(intel_subgroup_local_block_io)
        CASE(intel_dot_accumulate)

        CASE(intel_global_float_atomics)
        CASE(intel_subgroup_matrix_multiply_accumulate)
        CASE(intel_subgroup_split_matrix_multiply_accumulate)
        CASE(intel_variable_eu_thread_count)
        CASE(future_bf16_cvt)
        default: return nullptr;
    }
#undef CASE
}

struct runtime_version_t {
    int major;
    int minor;
    int build;

    runtime_version_t(int major = 0, int minor = 0, int build = 0)
        : major {major}, minor {minor}, build {build} {}

    bool operator==(const runtime_version_t &other) const {
        return (major == other.major) && (minor == other.minor)
                && (build == other.build);
    }

    bool operator!=(const runtime_version_t &other) const {
        return !(*this == other);
    }

    bool operator<(const runtime_version_t &other) const {
        if (major < other.major) return true;
        if (major > other.major) return false;
        if (minor < other.minor) return true;
        if (minor > other.minor) return false;
        return (build < other.build);
    }

    bool operator>(const runtime_version_t &other) const {
        return (other < *this);
    }

    bool operator<=(const runtime_version_t &other) const {
        return !(*this > other);
    }

    bool operator>=(const runtime_version_t &other) const {
        return !(*this < other);
    }

    status_t set_from_string(const char *s) {
        int i_major = 0, i = 0;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_minor = ++i;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_build = ++i;

        major = atoi(&s[i_major]);
        minor = atoi(&s[i_minor]);
        build = atoi(&s[i_build]);

        return status::success;
    }

    std::string str() const {
        return utils::format("%d.%d.%d", major, minor, build);
    }
};

// Needed workaround for future HW extensions
uint64_t get_future_extensions(compute::gpu_arch_t gpu_arch);

struct device_info_t {
public:
    virtual ~device_info_t() = default;

    status_t init(engine_t *engine) {
        CHECK(init_device_name(engine));
        CHECK(init_arch(engine));
        CHECK(init_runtime_version(engine));
        CHECK(init_extensions(engine));
        CHECK(init_attributes(engine));

        CHECK(init_attributes_common(engine));
        return status::success;
    }

    bool has(device_ext_t ext) const { return extensions_ & (uint64_t)ext; }
    gpu_arch_t gpu_arch() const { return gpu_arch_; }
    int stepping_id() const { return stepping_id_; }
    int max_eus_per_wg() const { return max_eus_per_wg_; }
    int eu_count() const { return eu_count_; }
    int hw_threads() const { return hw_threads_[0]; }
    int hw_threads(bool large_grf_mode) const {
        return hw_threads_[large_grf_mode ? 1 : 0];
    }
    size_t llc_cache_size() const { return llc_cache_size_; }

    const runtime_version_t &runtime_version() const {
        return runtime_version_;
    }
    const std::string &name() const { return name_; }

    bool mayiuse_ngen_kernels(engine_t *engine);

    bool mayiuse_non_uniform_work_groups() const {
        return mayiuse_non_uniform_work_groups_;
    }

    bool mayiuse_sub_group(int size) const;

protected:
    virtual status_t init_device_name(engine_t *engine) = 0;
    virtual status_t init_arch(engine_t *engine) = 0;
    virtual status_t init_runtime_version(engine_t *engine) = 0;
    virtual status_t init_extensions(engine_t *engine) = 0;
    virtual status_t init_attributes(engine_t *engine) = 0;

    compute::gpu_arch_t gpu_arch_ = compute::gpu_arch_t::unknown;
    int stepping_id_ = 0;

    std::string name_;
    runtime_version_t runtime_version_;

    // total number of hardware threads:
    // [0] - default mode
    // [1] - large GRF mode
    int32_t hw_threads_[2] = {0, 0};
    int32_t eu_count_ = 0;
    int32_t max_eus_per_wg_ = 0;
    size_t llc_cache_size_ = 0;

    // extensions_ and gpu_arch_ describe effective extensions and GPU architecture.
    uint64_t extensions_ = 0;

private:
    status_t init_attributes_common(engine_t *engine);

    bool mayiuse_ngen_kernels_ = false;
    bool checked_ngen_kernels_ = false;

    bool mayiuse_non_uniform_work_groups_ = false;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
