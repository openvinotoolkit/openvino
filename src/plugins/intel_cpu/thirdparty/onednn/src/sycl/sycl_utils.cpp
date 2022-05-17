/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "sycl/sycl_utils.hpp"

#include "sycl/level_zero_utils.hpp"
#include <CL/sycl/backend/opencl.hpp>

#ifdef DNNL_SYCL_CUDA
// Do not include sycl_cuda_utils.hpp because it's intended for use in
// gpu/nvidia directory only.

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {
bool compare_cuda_devices(
        const cl::sycl::device &lhs, const cl::sycl::device &rhs);
}
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

namespace dnnl {
namespace impl {
namespace sycl {

backend_t get_sycl_gpu_backend() {
    // Create default GPU device and query its backend (assumed as default)
    static backend_t default_backend = []() {
        const backend_t fallback = backend_t::opencl;

        const auto gpu_type = cl::sycl::info::device_type::gpu;
        if (cl::sycl::device::get_devices(gpu_type).empty()) return fallback;

        cl::sycl::device dev {cl::sycl::gpu_selector {}};
        backend_t backend = get_sycl_backend(dev);

#if !defined(DNNL_WITH_LEVEL_ZERO)
        if (backend == backend_t::level0) backend = fallback;
#endif

        return backend;
    }();

    return default_backend;
}

bool are_equal(const cl::sycl::device &lhs, const cl::sycl::device &rhs) {
    auto lhs_be = get_sycl_backend(lhs);
    auto rhs_be = get_sycl_backend(rhs);
    if (lhs_be != rhs_be) return false;

    // Only one host device exists.
    if (lhs_be == backend_t::host) return true;

    if (lhs_be == backend_t::opencl) {
        // Use wrapper objects to avoid memory leak.
        auto lhs_ocl_dev = gpu::ocl::make_ocl_wrapper(lhs.get());
        auto rhs_ocl_dev = gpu::ocl::make_ocl_wrapper(rhs.get());
        return lhs_ocl_dev == rhs_ocl_dev;
    }

    if (lhs_be == backend_t::level0) { return compare_ze_devices(lhs, rhs); }

#ifdef DNNL_SYCL_CUDA
    if (lhs_be == backend_t::nvidia) {
        return gpu::nvidia::compare_cuda_devices(lhs, rhs);
    }
#endif

    assert(!"not expected");
    return false;
}

device_id_t sycl_device_id(const cl::sycl::device &dev) {
    if (dev.is_host())
        return std::make_tuple(static_cast<int>(backend_t::host), 0, 0);

    device_id_t device_id
            = device_id_t {static_cast<int>(backend_t::unknown), 0, 0};
    switch (get_sycl_backend(dev)) {
        case backend_t::opencl: {
            auto ocl_device = gpu::ocl::make_ocl_wrapper(dev.get());
            device_id = std::make_tuple(static_cast<int>(backend_t::opencl),
                    reinterpret_cast<uint64_t>(ocl_device.get()), 0);
            break;
        }
        case backend_t::level0: {
#if defined(DNNL_WITH_LEVEL_ZERO)
            device_id = std::tuple_cat(
                    std::make_tuple(static_cast<int>(backend_t::level0)),
                    get_device_uuid(dev));
#else
            assert(!"unreachable");
#endif
            break;
        }
        case backend_t::unknown: assert(!"unknown backend"); break;
        default: assert(!"unreachable");
    }
    assert(std::get<0>(device_id) != static_cast<int>(backend_t::unknown));
    return device_id;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
