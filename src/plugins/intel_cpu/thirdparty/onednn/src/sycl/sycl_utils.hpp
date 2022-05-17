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

#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#include <vector>
#include <CL/sycl.hpp>

namespace dnnl {
namespace impl {
namespace sycl {

using buffer_u8_t = cl::sycl::buffer<uint8_t, 1>;

inline cl::sycl::nd_range<3> to_sycl_nd_range(
        const gpu::compute::nd_range_t &range) {
    auto *local_range = range.local_range();
    auto *global_range = range.global_range();

    auto sycl_global_range = cl::sycl::range<3>(
            global_range[2], global_range[1], global_range[0]);

    if (!local_range) {
        assert(!"not expected");
        return cl::sycl::nd_range<3>(
                sycl_global_range, cl::sycl::range<3>(1, 1, 1));
    }

    auto sycl_local_range = cl::sycl::range<3>(
            local_range[2], local_range[1], local_range[0]);
    return cl::sycl::nd_range<3>(sycl_global_range, sycl_local_range);
}

// Automatically use codeplay_host_task if it is supported by compiler,
// otherwise fall back to single_task.
template <typename K, typename H, typename F>
inline auto host_task_impl(H &cgh, F f, int)
        -> decltype(cgh.codeplay_host_task(f)) {
    cgh.codeplay_host_task(f);
}

template <typename K, typename H, typename F>
inline void host_task_impl(H &cgh, F f, long) {
    cgh.template single_task<K>(f);
}

template <typename K, typename H, typename F>
inline void host_task(H &cgh, F f) {
    // Third argument is 0 (int) which prefers the
    // run_on_host_intel option if both are available.
    host_task_impl<K>(cgh, f, 0);
}

enum class backend_t { unknown, host, level0, opencl, nvidia };

inline std::string to_string(backend_t backend) {
    switch (backend) {
        case backend_t::host: return "Host";
        case backend_t::level0: return "Level Zero";
        case backend_t::opencl: return "OpenCL";
        case backend_t::nvidia: return "Nvidia";
        default: return "Unknown";
    }
}

backend_t get_sycl_gpu_backend();

inline backend_t get_sycl_backend(const cl::sycl::device &dev) {
    if (dev.is_host()) return backend_t::host;

    auto plat = dev.get_platform();
    std::string plat_name = plat.get_info<cl::sycl::info::platform::name>();
    if (plat_name.find("OpenCL") != std::string::npos) return backend_t::opencl;
    if (plat_name.find("NVIDIA") != std::string::npos) return backend_t::nvidia;

    if (plat_name.find("Level-Zero") != std::string::npos)
        return backend_t::level0;

    return backend_t::unknown;
}

bool are_equal(const cl::sycl::device &lhs, const cl::sycl::device &rhs);
device_id_t sycl_device_id(const cl::sycl::device &dev);

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
