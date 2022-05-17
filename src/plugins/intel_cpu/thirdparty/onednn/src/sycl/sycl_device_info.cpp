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

#include <CL/sycl/backend/opencl.hpp>

#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_utils.hpp"

#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_gpu_hw_info.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_device_info_t::init_arch(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();

    // skip cpu engines
    if (!device.is_gpu()) return status::success;

    // skip other vendors
    const int intel_vendor_id = 0x8086;
    auto vendor_id = device.get_info<cl::sycl::info::device::vendor_id>();
    if (vendor_id != intel_vendor_id) return status::success;

    backend_t be = get_sycl_backend(device);
    if (be == backend_t::opencl) {
        cl_int err = CL_SUCCESS;

        auto ocl_dev_wrapper = gpu::ocl::make_ocl_wrapper(device.get());

        auto ocl_dev = ocl_dev_wrapper.get();
        auto ocl_ctx_wrapper = gpu::ocl::make_ocl_wrapper(
                clCreateContext(nullptr, 1, &ocl_dev, nullptr, nullptr, &err));
        OCL_CHECK(err);

        gpu::ocl::init_gpu_hw_info(
                ocl_dev_wrapper, ocl_ctx_wrapper, gpu_arch_, stepping_id_);
    } else if (be == backend_t::level0) {
        // TODO: add support for L0 binary ngen check
        // XXX: query from ocl_engine for now
        gpu::ocl::ocl_engine_factory_t f(engine_kind::gpu);

        engine_t *engine;
        CHECK(f.engine_create(&engine, 0));

        std::unique_ptr<gpu::compute::compute_engine_t, engine_deleter_t>
                compute_engine(
                        utils::downcast<gpu::compute::compute_engine_t *>(
                                engine));

        auto *dev_info = compute_engine->device_info();
        gpu_arch_ = dev_info->gpu_arch();
    } else {
        assert(!"not_expected");
    }

    // XXX: temporary WA for different Xe_HP devices
    if (gpu_arch_ == gpu::compute::gpu_arch_t::xe_hpg
            && !device.has_extension(
                    ext2cl_str(gpu::compute::device_ext_t::khr_fp64)))
        gpu_arch_ = gpu::compute::gpu_arch_t::xe_hpg;
    return status::success;
}

status_t sycl_device_info_t::init_device_name(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    name_ = device.get_info<cl::sycl::info::device::name>();
    return status::success;
}

status_t sycl_device_info_t::init_runtime_version(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    auto driver_version
            = device.get_info<cl::sycl::info::device::driver_version>();

    if (runtime_version_.set_from_string(driver_version.c_str())
            != status::success) {
        runtime_version_.major = 0;
        runtime_version_.minor = 0;
        runtime_version_.build = 0;
    }

    return status::success;
}

status_t sycl_device_info_t::init_extensions(engine_t *engine) {
    using namespace gpu::compute;

    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    std::string extension_string;
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        const char *s_ext = ext2cl_str((device_ext_t)i_ext);
        if (s_ext && device.has_extension(s_ext)) {
            extension_string += std::string(s_ext) + " ";
            extensions_ |= i_ext;
        }
    }

    // Handle future extensions, not yet supported by the DPC++ API
    extensions_ |= (uint64_t)get_future_extensions(gpu_arch());

    return status::success;
}

status_t sycl_device_info_t::init_attributes(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    eu_count_ = device.get_info<cl::sycl::info::device::max_compute_units>();
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
