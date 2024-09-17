// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <level_zero/ze_api.h>
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/device.hpp"

#include <memory>

namespace cldnn {
namespace ze {

class ze_engine : public engine {
public:
    ze_engine(const device::ptr dev, runtime_types runtime_type);
    engine_types type() const override { return engine_types::ze; };
    runtime_types runtime_type() const override { return runtime_types::ze; };

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) override;
    memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) override;
    bool is_the_same_buffer(const memory& mem1, const memory& mem2) override;
    bool check_allocatable(const layout& layout, allocation_type type) override;

    void* get_user_context() const override;

    allocation_type get_default_allocation_type() const override { return allocation_type::usm_device; }
    allocation_type detect_usm_allocation_type(const void* memory) const override;

    const ze_context_handle_t get_context() const;
    const ze_driver_handle_t get_driver() const;
    const ze_device_handle_t get_device() const;

    stream_ptr create_stream(const ExecutionConfig& config) const override;
    stream_ptr create_stream(const ExecutionConfig& config, void *handle) const override;
    stream& get_service_stream() const override;

    kernel::ptr prepare_kernel(const kernel::ptr kernel) const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    void create_onednn_engine(const ExecutionConfig& config) override;
    // Returns onednn engine object which shares device and context with current engine
    dnnl::engine& get_onednn_engine() const override;
#endif

    static std::shared_ptr<cldnn::engine> create(const device::ptr device, runtime_types runtime_type);

private:
    std::unique_ptr<stream> _service_stream;

#ifdef ENABLE_ONEDNN_FOR_GPU
    std::mutex onednn_mutex;
    std::shared_ptr<dnnl::engine> _onednn_engine;
#endif
};

}  // namespace ze
}  // namespace cldnn
