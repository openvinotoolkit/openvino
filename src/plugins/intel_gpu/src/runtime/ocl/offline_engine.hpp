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

#include <memory>

namespace cldnn {
namespace ocl {

// HW-free engine for offline compile-only. It carries a fabricated device_info via
// offline_device and creates NO cl::Context. It reports engine_types::ocl so the compile passes select
// the ocl impl path. Only the three methods the compile-only passes actually reach are functional:
//   - create_stream()          -> a no-op offline_stream (stored as program::_stream, never enqueued)
//   - create_kernel_builder()  -> offline_kernel_builder (stored by kernels_cache; offline path never
//                                 calls build_kernels, it emits ocloc placeholders)
//   - get_device_info()        -> base engine returns _device->get_info() (the fabricated info)
// Every other (memory/context/usm) method throws: reaching it means a real network is being built or
// executed on the compile-only engine, which must not happen (offline compile builds no network).
class offline_engine : public engine {
public:
    explicit offline_engine(const device::ptr dev) : engine(dev) {
        // Provide a service stream so any get_service_stream() call does not dereference null.
        _service_stream.reset(new offline_stream());
    }

    engine_types type() const override { return engine_types::ocl; }
    runtime_types runtime_type() const override { return runtime_types::ocl; }
    backend_types backend_type() const override { return backend_types::ocl; }

    stream_ptr create_stream(const ExecutionConfig& /*config*/) const override {
        return std::make_shared<offline_stream>();
    }
    stream_ptr create_stream(const ExecutionConfig& /*config*/, void* /*handle*/) const override {
        return std::make_shared<offline_stream>();
    }
    std::shared_ptr<kernel_builder> create_kernel_builder() const override {
        return std::make_shared<offline_kernel_builder>();
    }

    allocation_type get_default_allocation_type() const override { return allocation_type::usm_host; }

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
    memory_ptr create_mmap_hostbuffer(const void* /*mmapped_address*/, size_t /*data_size*/,
                                      allocation_type /*_allocation_type*/, const layout /*output_layout*/) override {
        OPENVINO_THROW("[GPU offline] engine::create_mmap_hostbuffer is not available on the compile-only engine");
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
    bool is_the_same_buffer(const memory& /*mem1*/, const memory& /*mem2*/) override {
        OPENVINO_THROW("[GPU offline] engine::is_the_same_buffer is not available on the compile-only engine");
    }
    // No real CL/L0 context exists. Returning null (rather than throwing) lets RemoteContextImpl::
    // init_properties() store a null ocl_context handle harmlessly (compile-only never uses it).
    void* get_user_context(runtime_types /*rt_type*/) const override { return nullptr; }
    allocation_type detect_usm_allocation_type(const void* /*memory*/) const override {
        OPENVINO_THROW("[GPU offline] engine::detect_usm_allocation_type is not available on the compile-only engine");
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    void create_onednn_engine(const ExecutionConfig& /*config*/) override {
        OPENVINO_THROW("[GPU offline] engine::create_onednn_engine is not available on the compile-only engine");
    }
#endif
};

}  // namespace ocl
}  // namespace cldnn
