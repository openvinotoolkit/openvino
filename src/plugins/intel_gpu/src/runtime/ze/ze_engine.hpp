// Copyright (C) 2016-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/stream.hpp"
#include "ze_device.hpp"

#include <memory>
#include <set>
#include <vector>
#include <utility>
#include <string>

namespace cldnn {
namespace ze {

class ze_engine : public engine {
public:
    ze_engine(const device::ptr dev, runtime_types runtime_type, const engine_configuration& conf,
                                                        const InferenceEngine::ITaskExecutor::Ptr);
    engine_types type() const override { return engine_types::ze; };
    runtime_types runtime_type() const override { return runtime_types::ze; };

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) override;
    memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) override;
    bool is_the_same_buffer(const memory& mem1, const memory& mem2) override;

    void* get_user_context() const override;

    allocation_type get_default_allocation_type() const override { return allocation_type::usm_host; }

    const ze_context_handle_t get_context() const;
    const ze_driver_handle_t get_driver() const;
    const ze_device_handle_t get_device() const;

    stream_ptr create_stream() const override;
    stream_ptr create_stream(void *handle) const override;
    stream& get_program_stream() const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    /// Returns onednn engine object which shares device and context with current engine
    dnnl::engine& get_onednn_engine() const override;
#endif

    static std::shared_ptr<cldnn::engine> create(const device::ptr device, runtime_types runtime_type,
                                    const engine_configuration& configuration, const InferenceEngine::ITaskExecutor::Ptr task_executor);

private:
    std::unique_ptr<stream> _program_stream;

#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::engine> _onednn_engine;
#endif
};

}  // namespace ze
}  // namespace cldnn
