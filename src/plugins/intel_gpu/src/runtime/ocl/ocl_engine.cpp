// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_engine.hpp"
#include "ocl_common.hpp"
#include "ocl_memory.hpp"
#include "ocl_stream.hpp"
#include "ocl_engine_factory.hpp"
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

// static class memebers - pointers to dynamically obtained OpenCL extension functions
cl::PFN_clEnqueueAcquireMediaSurfacesINTEL cl::SharedSurfLock::pfn_acquire = NULL;
cl::PFN_clEnqueueReleaseMediaSurfacesINTEL cl::SharedSurfLock::pfn_release = NULL;
cl::PFN_clCreateFromMediaSurfaceINTEL cl::ImageVA::pfn_clCreateFromMediaSurfaceINTEL = NULL;
#ifdef _WIN32
cl::PFN_clCreateFromD3D11Buffer cl::BufferDX::pfn_clCreateFromD3D11Buffer = NULL;
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include "intel_gpu/runtime/file_util.hpp"
#endif

namespace cldnn {
namespace ocl {

OPENVINO_SUPPRESS_DEPRECATED_START
ocl_error::ocl_error(cl::Error const& err)
    : ov::Exception("[GPU] " + std::string(err.what()) + std::string(", error code: ") + std::to_string(err.err())) {}
OPENVINO_SUPPRESS_DEPRECATED_END

ocl_engine::ocl_engine(const device::ptr dev, runtime_types runtime_type)
    : engine(dev) {
    OPENVINO_ASSERT(runtime_type == runtime_types::ocl, "[GPU] Invalid runtime type specified for OCL engine. Only OCL runtime is supported");

    auto casted = dynamic_cast<ocl_device*>(dev.get());
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type passed to ocl engine");
    casted->get_device().getInfo(CL_DEVICE_EXTENSIONS, &_extensions);

    _usm_helper.reset(new cl::UsmHelper(get_cl_context(), get_cl_device(), use_unified_shared_memory()));
    _service_stream.reset(new ocl_stream(*this, ExecutionConfig()));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void ocl_engine::create_onednn_engine(const ExecutionConfig& config) {
    const std::lock_guard<std::mutex> lock(onednn_mutex);
    OPENVINO_ASSERT(_device->get_info().vendor_id == INTEL_VENDOR_ID, "[GPU] OneDNN engine can be used for Intel GPUs only");
    if (!_onednn_engine) {
        auto casted = std::dynamic_pointer_cast<ocl_device>(_device);
        OPENVINO_ASSERT(casted, "[GPU] Invalid device type stored in ocl_engine");

        std::string cache_dir = config.get_property(ov::cache_dir);
        if (cache_dir.empty()) {
            _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ocl_interop::make_engine(casted->get_device().get(), casted->get_context().get()));
        } else {
            // Use cached blob
            auto path = cache_dir;
            if (path.back() != '/' && path.back() != '\\') {
                path += "/";
            }

            auto blob_id = dnnl::ocl_interop::get_engine_cache_blob_id(casted->get_device().get());
            if (blob_id.empty()) {
                // Create engine without cache_blob
                _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ocl_interop::make_engine(casted->get_device().get(), casted->get_context().get()));
                return;
            }

            std::string id_str(blob_id.begin(), blob_id.end());
            size_t hash = std::hash<std::string>()(id_str);
            path = path + std::to_string(hash) + ".onednn.cl_cache";

            auto onednn_cache_blob = ov::util::load_binary(path);
            if (onednn_cache_blob.empty()) {
                _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ocl_interop::make_engine(casted->get_device().get(), casted->get_context().get()));

                onednn_cache_blob = dnnl::ocl_interop::get_engine_cache_blob(*_onednn_engine);
                ov::intel_gpu::save_binary(path, onednn_cache_blob);
            } else {
                _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ocl_interop::make_engine(casted->get_device().get(), casted->get_context().get(),
                                                                                onednn_cache_blob));
            }
        }
    }
}

dnnl::engine& ocl_engine::get_onednn_engine() const {
    OPENVINO_ASSERT(_onednn_engine, "[GPU] Can't get onednn engine handle as it was not initialized. Please check that create_onednn_engine() was called");
    return *_onednn_engine;
}
#endif

const cl::Context& ocl_engine::get_cl_context() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    OPENVINO_ASSERT(cl_device, "[GPU] Invalid device type for ocl_engine");
    return cl_device->get_context();
}

const cl::Device& ocl_engine::get_cl_device() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    OPENVINO_ASSERT(cl_device, "[GPU] Invalid device type for ocl_engine");
    return cl_device->get_device();
}

const cl::UsmHelper& ocl_engine::get_usm_helper() const {
    return *_usm_helper;
}

allocation_type ocl_engine::detect_usm_allocation_type(const void* memory) const {
    return use_unified_shared_memory() ? ocl::gpu_usm::detect_allocation_type(this, memory)
                                       : allocation_type::unknown;
}

bool ocl_engine::check_allocatable(const layout& layout, allocation_type type) {
    OPENVINO_ASSERT(supports_allocation(type) || type == allocation_type::cl_mem, "[GPU] Unsupported allocation type: ", type);

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

memory::ptr ocl_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate memory for dynamic layout");

    check_allocatable(layout, type);

    try {
        memory::ptr res = nullptr;
        if (layout.format.is_image_2d()) {
            res = std::make_shared<ocl::gpu_image2d>(this, layout);
        } else if (type == allocation_type::cl_mem) {
            res = std::make_shared<ocl::gpu_buffer>(this, layout);
        } else {
            res = std::make_shared<ocl::gpu_usm>(this, layout, type);
        }

        if (reset || res->is_memory_reset_needed(layout)) {
            auto ev = res->fill(get_service_stream());
            if (ev) {
                get_service_stream().wait_for_events({ev});
            }
        }

        return res;
    } catch (const cl::Error& clErr) {
        switch (clErr.err()) {
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            case CL_OUT_OF_RESOURCES:
            case CL_OUT_OF_HOST_MEMORY:
            case CL_INVALID_BUFFER_SIZE:
                OPENVINO_THROW("[GPU] out of GPU resources");
            default:
                OPENVINO_THROW("[GPU] buffer allocation failed");
        }
    }
}

memory::ptr ocl_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to reinterpret buffer allocated by a different engine");
    OPENVINO_ASSERT(new_layout.format.is_image() == memory.get_layout().format.is_image(),
                    "[GPU] trying to reinterpret between image and non-image layouts. Current: ",
                    memory.get_layout().format.to_string(), " Target: ", new_layout.format.to_string());

    try {
        if (new_layout.format.is_image_2d()) {
           return std::make_shared<ocl::gpu_image2d>(this,
                                     new_layout,
                                     reinterpret_cast<const ocl::gpu_image2d&>(memory).get_buffer(),
                                     memory.get_mem_tracker());
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            return std::make_shared<ocl::gpu_usm>(this,
                                     new_layout,
                                     reinterpret_cast<const ocl::gpu_usm&>(memory).get_buffer(),
                                     memory.get_allocation_type(),
                                     memory.get_mem_tracker());
        } else {
           return std::make_shared<ocl::gpu_buffer>(this,
                                    new_layout,
                                    reinterpret_cast<const ocl::gpu_buffer&>(memory).get_buffer(),
                                    memory.get_mem_tracker());
        }
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
}

memory::ptr ocl_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
   try {
        if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_image) {
            cl::Image2D img(static_cast<cl_mem>(params.mem), true);
            return std::make_shared<ocl::gpu_image2d>(this, new_layout, img, nullptr);
        } else if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_vasurface) {
            return std::make_shared<ocl::gpu_media_buffer>(this, new_layout, params);
#ifdef _WIN32
        } else if (params.mem_type == shared_mem_type::shared_mem_dxbuffer) {
            return std::make_shared<ocl::gpu_dx_buffer>(this, new_layout, params);
#endif
        } else if (params.mem_type == shared_mem_type::shared_mem_buffer) {
            cl::Buffer buf(static_cast<cl_mem>(params.mem), true);
            auto actual_mem_size = buf.getInfo<CL_MEM_SIZE>();
            auto requested_mem_size = new_layout.bytes_count();
            OPENVINO_ASSERT(actual_mem_size >= requested_mem_size,
                            "[GPU] shared buffer has smaller size (", actual_mem_size,
                            ") than specified layout (", requested_mem_size, ")");
            return std::make_shared<ocl::gpu_buffer>(this, new_layout, buf, nullptr);
        } else if (params.mem_type == shared_mem_type::shared_mem_usm) {
            cl::UsmMemory usm_buffer(get_usm_helper(), params.mem);
            auto actual_mem_size = get_usm_helper().get_usm_allocation_size(usm_buffer.get());
            auto requested_mem_size = new_layout.bytes_count();
            OPENVINO_ASSERT(actual_mem_size >= requested_mem_size,
                            "[GPU] shared USM buffer has smaller size (", actual_mem_size,
                            ") than specified layout (", requested_mem_size, ")");
            return std::make_shared<ocl::gpu_usm>(this, new_layout, usm_buffer, nullptr);
        } else {
            OPENVINO_THROW("[GPU] unknown shared object fromat or type");
        }
    }
    catch (const cl::Error& clErr) {
        switch (clErr.err()) {
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY:
        case CL_INVALID_BUFFER_SIZE:
            OPENVINO_THROW("[GPU] out of GPU resources");
        default:
            OPENVINO_THROW("[GPU] buffer allocation failed");
        }
    }
}

bool ocl_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;

    if (!memory_capabilities::is_usm_type(mem1.get_allocation_type()))
        return (reinterpret_cast<const ocl::gpu_buffer&>(mem1).get_buffer() ==
                reinterpret_cast<const ocl::gpu_buffer&>(mem2).get_buffer());
    else
        return (reinterpret_cast<const ocl::gpu_usm&>(mem1).get_buffer() ==
                reinterpret_cast<const ocl::gpu_usm&>(mem2).get_buffer());
}

void* ocl_engine::get_user_context() const {
    auto& cl_device = downcast<ocl_device>(*_device);
    return static_cast<void*>(cl_device.get_context().get());
}

bool ocl_engine::extension_supported(std::string extension) const {
    return _extensions.find(extension) != std::string::npos;
}

stream::ptr ocl_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<ocl_stream>(*this, config);
}

stream::ptr ocl_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    return std::make_shared<ocl_stream>(*this, config, handle);
}

stream& ocl_engine::get_service_stream() const {
    return *_service_stream;
}

std::shared_ptr<cldnn::engine> ocl_engine::create(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<ocl::ocl_engine>(device, runtime_type);
}

std::shared_ptr<cldnn::engine> create_ocl_engine(const device::ptr device, runtime_types runtime_type) {
    return ocl_engine::create(device, runtime_type);
}

}  // namespace ocl
}  // namespace cldnn
