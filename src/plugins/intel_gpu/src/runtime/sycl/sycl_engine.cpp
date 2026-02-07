// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "sycl/sycl_kernel.hpp"
#include "sycl_common.hpp"
#include "sycl_memory.hpp"
#include "sycl_stream.hpp"
#include "sycl_engine_factory.hpp"
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>
#include "ocl/ocl_kernel.hpp"  // for testing purposes

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif









#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include "intel_gpu/runtime/file_util.hpp"
#endif

namespace cldnn {
namespace sycl {






sycl_engine::sycl_engine(const device::ptr dev, runtime_types runtime_type)
    : engine(dev) {
    OPENVINO_ASSERT(runtime_type == runtime_types::sycl, "[GPU] Invalid runtime type specified for SYCL engine. Only SYCL runtime is supported");

    auto casted = dynamic_cast<sycl_device*>(dev.get());
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type passed to sycl engine");
    try {
        _extensions = casted->get_device().get_info<::sycl::info::device::aspects>();
    } catch (const ::sycl::exception& e) {
        OPENVINO_THROW("[GPU] SYCL Engine initialization failed: ", e.what());
    }

    _service_stream.reset(new sycl_stream(*this, ExecutionConfig()));
}

backend_types sycl_engine::backend_type() const {
    auto backend = get_sycl_device().get_backend();
    switch (backend) {
        case ::sycl::backend::opencl: return backend_types::ocl;
        case ::sycl::backend::ext_oneapi_hip: return backend_types::hip;
        case ::sycl::backend::ext_oneapi_cuda: return backend_types::cuda;
        case ::sycl::backend::ext_oneapi_level_zero: return backend_types::l0;
        default:
            OPENVINO_THROW("[GPU] Unsupported SYCL backend type: ", backend);
    }
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void sycl_engine::create_onednn_engine(const ExecutionConfig& config) {
    const std::lock_guard<std::mutex> lock(onednn_mutex);
    OPENVINO_ASSERT(_device->get_info().vendor_id == INTEL_VENDOR_ID, "[GPU] OneDNN engine can be used for Intel GPUs only");
    if (!_onednn_engine) {
        auto casted = std::dynamic_pointer_cast<sycl_device>(_device);
        OPENVINO_ASSERT(casted, "[GPU] Invalid device type stored in sycl_engine");

        _onednn_engine = std::make_shared<dnnl::engine>(dnnl::sycl_interop::make_engine(casted->get_device(), casted->get_context()));
    }
}

dnnl::engine& sycl_engine::get_onednn_engine() const {
    OPENVINO_ASSERT(_onednn_engine, "[GPU] Can't get onednn engine handle as it was not initialized. Please check that create_onednn_engine() was called");
    return *_onednn_engine;
}
#endif

const ::sycl::context& sycl_engine::get_sycl_context() const {
    auto sycl_device = std::dynamic_pointer_cast<sycl::sycl_device>(_device);
    OPENVINO_ASSERT(sycl_device, "[GPU] Invalid device type for sycl_engine");
    return sycl_device->get_context();
}

const ::sycl::device& sycl_engine::get_sycl_device() const {
    auto sycl_device = std::dynamic_pointer_cast<sycl::sycl_device>(_device);
    OPENVINO_ASSERT(sycl_device, "[GPU] Invalid device type for sycl_engine");
    return sycl_device->get_device();
}

// const ::sycl::UsmHelper& sycl_engine::get_usm_helper() const {
//     auto sycl_device = std::dynamic_pointer_cast<sycl::sycl_device>(_device);
//     OPENVINO_ASSERT(sycl_device, "[GPU] Invalid device type for sycl_engine");
//     return sycl_device->get_usm_helper();
// }

allocation_type sycl_engine::detect_usm_allocation_type(const void* memory) const {
    if (use_unified_shared_memory()) {
        OPENVINO_NOT_IMPLEMENTED;
        // TODO: implement
        //::sycl::gpu_usm::detect_allocation_type(this, memory);
    } else {
        return allocation_type::unknown;
    }
}

bool sycl_engine::check_allocatable(const layout& layout, allocation_type type) {
    OPENVINO_ASSERT(supports_allocation(type) || type == allocation_type::sycl_buffer, "[GPU] Unsupported allocation type: ", type);

    bool exceed_allocatable_mem_size = (layout.bytes_count() > get_device_info().max_alloc_mem_size);

    // When dynamic shape upper bound makes bigger buffer, then return false.
    if (exceed_allocatable_mem_size && layout.is_dynamic()) {
        OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
        return false;
    }

    OPENVINO_ASSERT(!exceed_allocatable_mem_size,
                    "[GPU] Exceeded max size of memory object allocation: ",
                    "requested ", layout.bytes_count(), " bytes, "
                    "but max alloc size supported by device is ", get_device_info().max_alloc_mem_size, " bytes.",
                    "Please try to reduce batch size or use lower precision.");

    auto used_mem = get_used_device_memory(allocation_type::usm_device) + get_used_device_memory(allocation_type::usm_host);
    auto exceed_available_mem_size = (layout.bytes_count() + used_mem > get_max_memory_size());

    // When dynamic shape upper bound makes bigger buffer, then return false.
    if (exceed_available_mem_size && layout.is_dynamic()) {
        OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
        return false;
    }

#ifdef __unix__
    // Prevent from being killed by Ooo Killer of Linux
    OPENVINO_ASSERT(!exceed_available_mem_size,
                    "[GPU] Exceeded max size of memory allocation: ",
                    "Required ", layout.bytes_count(), " bytes, already occupied : ", used_mem, " bytes, ",
                    "but available memory size is ", get_max_memory_size(), " bytes");
#else
    if (exceed_available_mem_size) {
        GPU_DEBUG_COUT << "[Warning] [GPU] Exceeded max size of memory allocation: " << "Required " << layout.bytes_count() << " bytes, already occupied : "
                       << used_mem << " bytes, but available memory size is " << get_max_memory_size() << " bytes" << std::endl;
        GPU_DEBUG_COUT << "Please note that performance might drop due to memory swap." << std::endl;
        return false;
    }
#endif

    return true;
}

memory::ptr sycl_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate memory for dynamic layout");

    check_allocatable(layout, type);

    try {
        memory::ptr res = nullptr;
        if (layout.format.is_image_2d()) {
            OPENVINO_NOT_IMPLEMENTED;
            //res = std::make_shared<sycl::gpu_image2d>(this, layout);
        } else if (type == allocation_type::sycl_buffer) {
            res = std::make_shared<sycl::gpu_buffer>(this, layout);
        } else if (type == allocation_type::cl_mem) {
            OPENVINO_THROW("[GPU] SYCL engine does not support allocation_type::cl_mem");
        } else {
            OPENVINO_NOT_IMPLEMENTED;
            // res = std::make_shared<sycl::gpu_usm>(this, layout, type);
        }

        if (reset || res->is_memory_reset_needed(layout)) {
            auto ev = res->fill(get_service_stream());
            if (ev) {
                get_service_stream().wait_for_events({ev});
            }
        }

        return res;
    } catch (const ::sycl::exception& e) {
        OPENVINO_THROW("[GPU] SYCL memory allocation failed: ", e.what());
    }
}

memory::ptr sycl_engine::create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to create a subbuffer from a buffer allocated by a different engine");
    try {
        if (new_layout.format.is_image_2d()) {
            OPENVINO_NOT_IMPLEMENTED;
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            OPENVINO_NOT_IMPLEMENTED;
        } else {
            return downcast<const sycl::gpu_buffer>(memory).create_subbuffer(new_layout, byte_offset);
        }
    } catch (const ::sycl::exception& e) {
        OPENVINO_THROW("[GPU] SYCL subbuffer creation failed: ", e.what());
    }
}
memory::ptr sycl_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to reinterpret buffer allocated by a different engine");
    OPENVINO_ASSERT(new_layout.format.is_image() == memory.get_layout().format.is_image(),
                    "[GPU] trying to reinterpret between image and non-image layouts. Current: ",
                    memory.get_layout().format.to_string(), " Target: ", new_layout.format.to_string());

    try {
        if (new_layout.format.is_image_2d()) {
            OPENVINO_NOT_IMPLEMENTED;
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            OPENVINO_NOT_IMPLEMENTED;
        } else {
            return downcast<const sycl::gpu_buffer>(memory).reinterpret(new_layout);
        }
    } catch (const ::sycl::exception& e) {
        OPENVINO_THROW("[GPU] SYCL buffer reinterpretation failed: ", e.what());
    }
}

memory::ptr sycl_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    try {
        if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_image) {
            OPENVINO_NOT_IMPLEMENTED;
        } else if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_vasurface) {
            OPENVINO_THROW("[GPU] SYCL runtime does not support shared_mem_vasurface for image_2d");
#ifdef _WIN32
        } else if (params.mem_type == shared_mem_type::shared_mem_dxbuffer) {
            OPENVINO_THROW("[GPU] SYCL runtime does not support shared_mem_dxbuffer");
#endif
        } else if (params.mem_type == shared_mem_type::shared_mem_buffer) {
            OPENVINO_NOT_IMPLEMENTED;
        } else if (params.mem_type == shared_mem_type::shared_mem_usm) {
            OPENVINO_NOT_IMPLEMENTED;
        } else {
            OPENVINO_THROW("[GPU] unknown shared object fromat or type");
        }
    } catch (const ::sycl::exception& e) {
        OPENVINO_THROW("[GPU] SYCL handle reinterpretation failed: ", e.what());
    }
}

bool sycl_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;

    if (!memory_capabilities::is_usm_type(mem1.get_allocation_type()))
        return (reinterpret_cast<const sycl::gpu_buffer&>(mem1).get_buffer() ==
                reinterpret_cast<const sycl::gpu_buffer&>(mem2).get_buffer());
    else
        OPENVINO_NOT_IMPLEMENTED;
        // return (reinterpret_cast<const sycl::gpu_usm&>(mem1).get_buffer() ==
        //         reinterpret_cast<const sycl::gpu_usm&>(mem2).get_buffer());
}

void* sycl_engine::get_user_context() const {
     auto& sycl_device = downcast<sycl::sycl_device>(*_device);
     return const_cast<void*>(static_cast<const void*>(&sycl_device.get_context()));
}

kernel::ptr sycl_engine::prepare_kernel(const kernel::ptr kernel) const {
    // TODO: remove ocl_kernel check
    // currently accept both sycl and ocl kernels for testing purposes
    if (dynamic_cast<const ocl::ocl_kernel*>(kernel.get())) {
        return kernel;
    }

    OPENVINO_ASSERT(downcast<const sycl::sycl_kernel>(kernel.get()) != nullptr);
    return kernel;
}

bool sycl_engine::extension_supported(::sycl::aspect extension) const {
    return std::find(_extensions.begin(), _extensions.end(), extension) != _extensions.end();
}

stream::ptr sycl_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<sycl_stream>(*this, config);
}

stream::ptr sycl_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    return std::make_shared<sycl_stream>(*this, config, handle);
}

stream& sycl_engine::get_service_stream() const {
    return *_service_stream;
}

std::shared_ptr<cldnn::engine> sycl_engine::create(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<sycl::sycl_engine>(device, runtime_type);
}

std::shared_ptr<cldnn::engine> create_sycl_engine(const device::ptr device, runtime_types runtime_type) {
    return sycl_engine::create(device, runtime_type);
}

}  // namespace sycl
}  // namespace cldnn
