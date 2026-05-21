// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "ze_kernel_builder.hpp"
#include "openvino/zero_api.hpp"
#include "ze_engine_factory.hpp"
#include "ze_common.hpp"
#include "ze_memory.hpp"
#include "ze_stream.hpp"
#include "ze_device.hpp"
#include "ze_kernel.hpp"
#include "ze_ocl_exporter.hpp"
#include "ze_ocl_importer.hpp"
#include <exception>
#include <vector>
#include <memory>
#include <stdexcept>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ze.hpp>
#endif
namespace cldnn {
namespace ze {

ze_engine::ze_engine(const device::ptr dev, runtime_types runtime_type)
    : engine(dev) {
    OPENVINO_ASSERT(runtime_type == runtime_types::ze, "[GPU] Invalid runtime type specified for ZE engine. Only ZE runtime is supported");

    auto casted = dynamic_cast<ze_device*>(dev.get());
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type passed to ze engine");

    _service_stream.reset(new ze_stream(*this, ExecutionConfig()));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void ze_engine::create_onednn_engine(const ExecutionConfig& config) {
    const std::lock_guard<std::mutex> lock(onednn_mutex);
    OPENVINO_ASSERT(_device->get_info().vendor_id == INTEL_VENDOR_ID, "[GPU] OneDNN engine can be used for Intel GPUs only");
    if (!_onednn_engine) {
        _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ze_interop::make_engine(
            get_driver().get_ze_handle(),
            get_device().get_ze_handle(),
            get_context().get_ze_handle()
        ));
    }
}
#endif
const ze_driver_resource& ze_engine::get_driver() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_driver();
}

const ze_context_resource& ze_engine::get_context() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_context();
}

const ze_device_resource& ze_engine::get_device() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_device();
}

allocation_type ze_engine::detect_usm_allocation_type(const void* memory) const {
    return ze::gpu_usm::detect_allocation_type(this, memory);
}

memory::ptr ze_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate memory for dynamic layout");

    check_allocatable(layout, type);

    try {
        memory::ptr res;
        if (layout.format.is_image_2d()) {
            res = std::make_shared<ze::gpu_image2d>(this, layout);
        } else if (memory_capabilities::is_usm_type(type) || type == allocation_type::cl_mem){
            res = std::make_shared<ze::gpu_usm>(this, layout, type);
        } else {
            OPENVINO_THROW("[GPU] Unsupported allocation type: ", type);
        }

        if (reset || res->is_memory_reset_needed(layout)) {
            auto ev = res->fill(get_service_stream());
            if (ev) {
                get_service_stream().wait_for_events({ev});
            }
        }

        return res;
    } catch (const std::exception& e) {
        OPENVINO_THROW("[GPU] Failed to allocate memory: ", e.what());
    }
}

memory::ptr ze_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to reinterpret buffer allocated by a different engine");
    OPENVINO_ASSERT(new_layout.format.is_image() == memory.get_layout().format.is_image(),
                    "[GPU] trying to reinterpret between image and non-image layouts. Current: ",
                    memory.get_layout().format.to_string(), " Target: ", new_layout.format.to_string());

    bool from_memory_pool = memory.from_memory_pool;
    memory::ptr reinterpret_memory = nullptr;
    if (memory_capabilities::is_usm_type(memory.get_allocation_type())
        || memory.get_allocation_type() == allocation_type::cl_mem) {
        reinterpret_memory = std::make_shared<ze::gpu_usm>(this,
                                     new_layout,
                                     reinterpret_cast<const ze::gpu_usm&>(memory).get_resource(),
                                     memory.get_allocation_type(),
                                     memory.get_mem_tracker());
    } else if (new_layout.format.is_image_2d()) {
        reinterpret_memory = std::make_shared<ze::gpu_image2d>(this,
                                     new_layout,
                                     reinterpret_cast<const ze::gpu_image2d&>(memory).get_resource(),
                                     memory.get_mem_tracker());
    } else {
        OPENVINO_THROW("[GPU] Unexpected memory type for reinterpret_buffer");
    }
    reinterpret_memory->from_memory_pool = from_memory_pool;
    return reinterpret_memory;
}

memory::ptr ze_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    // Convert OCL handles from `params` to L0 and create memory objects
    if (params.mem_type == shared_mem_type::shared_mem_usm) {
        // USM memory does not need to be converted
        const auto &ctx = get_context();
        ov_ze_usm_handle usm_handle{ctx.get_ze_handle(), params.mem};
        bool is_shared = true;
        ze_usm_resource usm_res(usm_handle, is_shared);
        return std::make_shared<ze::gpu_usm>(this, new_layout, usm_res, nullptr);
    }  else if (params.mem_type == shared_mem_type::shared_mem_buffer) {
        const auto &ctx = get_context();
        auto ocl_buffer = static_cast<cl_mem>(params.mem);
        ze_ocl_importer<ocl_resource_type::mem_object, ze_resource_type::usm_memory> buffer_importer({ctx.get_ze_handle()});
        auto imported_buffer = buffer_importer(ocl_buffer);
        return std::make_shared<ze::gpu_usm>(this, new_layout, imported_buffer, allocation_type::cl_mem, nullptr);
    } else if (params.mem_type == shared_mem_type::shared_mem_image) {
        auto ocl_image = static_cast<cl_mem>(params.mem);
        ze_ocl_importer<ocl_resource_type::mem_object, ze_resource_type::image> image_importer;
        auto imported_image = image_importer(ocl_image);
        return std::make_shared<ze::gpu_image2d>(this, new_layout, imported_image, nullptr);
    } else {
        OPENVINO_THROW("[GPU] Unsupported shared memory type: ", params.mem_type);
    }
}

memory_ptr ze_engine::create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] Trying to create a subbuffer from a buffer allocated by a different engine");
    if (new_layout.format.is_image_2d()) {
        OPENVINO_NOT_IMPLEMENTED;
    }
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(memory.get_allocation_type()), "[GPU] Trying to create subbuffer for non usm memory");
    auto& new_buf = reinterpret_cast<const ze::gpu_usm&>(memory);
    auto ptr = new_buf.buffer_ptr();
    auto ctx = get_context();
    ov_ze_usm_handle usm_handle{ctx.get_ze_handle(), reinterpret_cast<uint8_t*>(ptr) + byte_offset};
    bool is_shared = true;
    ze_usm_resource usm_res(usm_handle, is_shared);
    return std::make_shared<ze::gpu_usm>(this,
                             new_layout,
                             usm_res,
                             memory.get_allocation_type(),
                             memory.get_mem_tracker());
}

bool ze_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;
    
    auto alloc_type = mem1.get_allocation_type();
    if (memory_capabilities::is_usm_type(mem1.get_allocation_type()) || alloc_type == allocation_type::cl_mem) {
        const auto &usm1 = downcast<const ze::gpu_usm>(mem1);
        const auto &usm2 = downcast<const ze::gpu_usm>(mem2);
        return usm1.buffer_ptr() == usm2.buffer_ptr();
    } else {
        const auto &img1 = downcast<const ze::gpu_image2d>(mem1);
        const auto &img2 = downcast<const ze::gpu_image2d>(mem2);
        return img1.get_resource().get_ze_handle() == img2.get_resource().get_ze_handle();
    }
    OPENVINO_THROW("[GPU] Unsupported memory type for buffer comparison");
}

std::shared_ptr<kernel_builder> ze_engine::create_kernel_builder() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return std::make_shared<ze_kernel_builder>(*casted);
}

void* ze_engine::get_user_context(runtime_types rt_type) const {
    auto ctx = get_context();
    if (rt_type == runtime_types::ze) {
        return ctx.get_ze_handle();
    } else if (rt_type == runtime_types::ocl) {
        auto &device = get_device();
        ze_ocl_exporter<ze_resource_type::context, ocl_resource_type::context> ctx_exporter({device});
        ctx_exporter(ctx);
        return ctx.get_ocl_handle<ocl_resource_type::context>();
    } else {
        OPENVINO_THROW("[GPU] ZE engine cannot provide context for ", rt_type);
    }
}

stream::ptr ze_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<ze_stream>(*this, config);
}

stream::ptr ze_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    cl_command_queue ocl_handle = static_cast<cl_command_queue>(handle);
    ze_ocl_importer<ocl_resource_type::command_queue, ze_resource_type::command_list> cmd_list_importer;
    auto ze_cmd_list = cmd_list_importer(ocl_handle);
    return std::make_shared<ze_stream>(*this, config, ze_cmd_list);
}

std::shared_ptr<cldnn::engine> ze_engine::create(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<ze::ze_engine>(device, runtime_type);
}

std::shared_ptr<cldnn::engine> create_ze_engine(const device::ptr device, runtime_types runtime_type) {
    return ze_engine::create(device, runtime_type);
}

}  // namespace ze
}  // namespace cldnn
