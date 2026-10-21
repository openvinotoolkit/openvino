// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "offline_device.hpp"
#include "offline_stream.hpp"
#include "offline_kernel_builder.hpp"
#include "offline_memory.hpp"
#include "openvino/core/except.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ocl.hpp>
#endif

#include <memory>

namespace cldnn {
namespace ocl {

// HW-free engine creating no cl::Context, reporting engine_types::ocl so the ocl impl path is chosen.
// Only stream/kernel_builder/device_info are functional; anything else means a network is being built.
class offline_engine : public engine {
public:
    explicit offline_engine(const device::ptr dev) : engine(dev) {
        // Provide a service stream so any get_service_stream() call does not dereference null.
        _service_stream.reset(new offline_stream());
    }

    engine_types type() const override { return engine_types::ocl; }
    runtime_types runtime_type() const override { return runtime_types::ocl; }
    backend_types backend_type() const override { return backend_types::ocl; }
    bool is_hw_free() const override { return true; }

    stream_ptr create_stream(const ExecutionConfig& /*config*/) const override {
        return std::make_shared<offline_stream>();
    }
    stream_ptr create_stream(const ExecutionConfig& /*config*/, void* /*handle*/) const override {
        return std::make_shared<offline_stream>();
    }
    std::shared_ptr<kernel_builder> create_kernel_builder() const override {
        return std::make_shared<offline_kernel_builder>();
    }

    allocation_type get_default_allocation_type() const override { return allocation_type::cl_mem; }

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool /*reset*/ = true) override {
        // Host-backed allocation (no cl::Context). Used for constant weights, which are memcpy'd in via
        // mem_lock and serialized into the blob. The buffer is zero-initialized so reset is a no-op.
        return std::make_shared<offline_memory>(this, layout, type);
    }
    memory_ptr reinterpret_handle(const layout& /*new_layout*/, shared_mem_params /*params*/) override {
        OPENVINO_THROW("[GPU offline] engine::reinterpret_handle is not available on the compile-only engine");
    }
    memory_ptr create_subbuffer(const memory& /*memory*/, const layout& /*new_layout*/, size_t /*byte_offset*/) override {
        OPENVINO_THROW("[GPU offline] engine::create_subbuffer is not available on the compile-only engine");
    }
    memory_ptr create_hostbuffer(void* /*cpu_address*/, size_t /*data_size*/,
                                 allocation_type /*_allocation_type*/, const layout /*output_layout*/) override {
        OPENVINO_THROW("[GPU offline] engine::create_hostbuffer is not available on the compile-only engine");
    }
    memory_ptr create_hostbuffer(const void* /*cpu_address*/, size_t /*data_size*/,
                                 allocation_type /*_allocation_type*/, const layout /*output_layout*/,
                                 bool /*host_read_only*/ = false) override {
        OPENVINO_THROW("[GPU offline] engine::create_hostbuffer is not available on the compile-only engine");
    }
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) override {
        // Return a view over the same host buffer with a new layout (e.g. constant reshaping in
        // prepare_primitive_fusing at compile time). Only offline_memory is produced by this engine.
        const auto* om = dynamic_cast<const offline_memory*>(&memory);
        OPENVINO_ASSERT(om, "[GPU offline] reinterpret_buffer expects offline_memory on the compile-only engine");
        return std::make_shared<offline_memory>(this, new_layout, memory.get_allocation_type(), om->shared_buffer());
    }
    memory_ptr import_buffer(const layout& /*layout*/, ov::intel_gpu::os_handle_param /*external_handle*/) override {
        OPENVINO_THROW("[GPU offline] engine::import_buffer is not available on the compile-only engine");
    }
    bool is_the_same_buffer(const memory& /*mem1*/, const memory& /*mem2*/) const override {
        OPENVINO_THROW("[GPU offline] engine::is_the_same_buffer is not available on the compile-only engine");
    }
    // No real CL/L0 context exists. Returning null (rather than throwing) lets RemoteContextImpl::
    // init_properties() store a null ocl_context handle harmlessly (compile-only never uses it).
    void* get_user_context(runtime_types /*rt_type*/) const override { return nullptr; }
    allocation_type detect_usm_allocation_type(const void* /*memory*/) const override {
        OPENVINO_THROW("[GPU offline] engine::detect_usm_allocation_type is not available on the compile-only engine");
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    // HW-free: oneDNN builds its engine from the device data captured during enumeration, so no
    // driver call is made. Such an engine serves primitive descriptor creation only; the primitives
    // themselves are JIT-compiled at import time, when a real device is available.
    void create_onednn_engine(const ExecutionConfig& /*config*/) override {
        if (_onednn_engine)
            return;

        const auto& onednn_device_info = static_cast<offline_device*>(_device.get())->onednn_device_info();
        OPENVINO_ASSERT(!onednn_device_info.empty(),
                        "[GPU offline] the device descriptor carries no oneDNN device data. Re-export it "
                        "from a machine with the target GPU using a build that has oneDNN enabled.");
        _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ocl_interop::make_offline_engine(onednn_device_info));
    }
#endif
};

}  // namespace ocl
}  // namespace cldnn
