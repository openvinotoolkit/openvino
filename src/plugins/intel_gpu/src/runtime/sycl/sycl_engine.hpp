// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "sycl_device.hpp"

#include <memory>
#include <set>
#include <vector>
#include <utility>
#include <string>

namespace cldnn {
namespace sycl {

class sycl_engine : public engine {
public:
    sycl_engine(const device::ptr dev, runtime_types runtime_type);
    engine_types type() const override { return engine_types::sycl; };
    runtime_types runtime_type() const override { return runtime_types::sycl; };
    backend_types backend_type() const override;

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) override;
    memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) override;
    memory_ptr create_subbuffer(const memory& memory, const layout& new_layout, size_t offset) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) override;
    bool is_the_same_buffer(const memory& mem1, const memory& mem2) override;

    void* get_user_context() const override;

    allocation_type get_default_allocation_type() const override { return allocation_type::sycl_buffer; }
    allocation_type detect_usm_allocation_type(const void* memory) const override;

    const ::sycl::context& get_sycl_context() const;
    const ::sycl::device& get_sycl_device() const;

    bool extension_supported(::sycl::aspect extension) const;

    stream_ptr create_stream(const ExecutionConfig& config) const override;
    stream_ptr create_stream(const ExecutionConfig& config, void *handle) const override;

    std::shared_ptr<kernel_builder> create_kernel_builder() const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    void create_onednn_engine(const ExecutionConfig& config) override;
#endif

    static std::shared_ptr<cldnn::engine> create(const device::ptr device, runtime_types runtime_type);
};

}  // namespace sycl
}  // namespace cldnn
