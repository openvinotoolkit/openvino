// Copyright (C) 2016-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "ocl_device.hpp"

#include <memory>
#include <set>
#include <vector>
#include <utility>
#include <string>

namespace cldnn {
namespace ocl {

class ocl_engine : public engine {
public:
    ocl_engine(const device::ptr dev, runtime_types runtime_type);
    engine_types type() const override { return engine_types::ocl; };
    runtime_types runtime_type() const override { return runtime_types::ocl; };

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) override;
    memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) override;
    bool is_the_same_buffer(const memory& mem1, const memory& mem2) override;
    bool check_allocatable(const layout& layout, allocation_type type) override;

    void* get_user_context() const override;

    allocation_type get_default_allocation_type() const override { return allocation_type::cl_mem; }
    allocation_type detect_usm_allocation_type(const void* memory) const override;

    const cl::Context& get_cl_context() const;
    const cl::Device& get_cl_device() const;
    const cl::UsmHelper& get_usm_helper() const;

    bool extension_supported(std::string extension) const;

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
    std::string _extensions;
    std::unique_ptr<stream> _service_stream;

#ifdef ENABLE_ONEDNN_FOR_GPU
    std::mutex onednn_mutex;
    std::shared_ptr<dnnl::engine> _onednn_engine;
#endif
};

}  // namespace ocl
}  // namespace cldnn
