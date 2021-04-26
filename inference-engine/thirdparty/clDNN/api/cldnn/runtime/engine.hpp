// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device.hpp"
#include "engine_configuration.hpp"
#include "event.hpp"
#include "memory_caps.hpp"
#include "memory_pool.hpp"
#include "layout.hpp"

#include <memory>
#include <set>
#include <vector>
#include <utility>
#include <string>

namespace cldnn {

class stream;

using memory_ptr = std::shared_ptr<memory>;
using stream_ptr = std::shared_ptr<stream>;

using primitive_id = std::string;

struct engine {
public:
    using ptr = std::unique_ptr<engine>();
    engine(const device::ptr device, const engine_configuration& configuration);

    virtual ~engine() = default;
    virtual engine_types type() const = 0;
    virtual runtime_types runtime_type() const = 0;

    memory_ptr get_memory_from_pool(const layout& layout,
                                    primitive_id,
                                    uint32_t network_id,
                                    std::set<primitive_id>,
                                    allocation_type type,
                                    bool reusable = true);

    memory_ptr attach_memory(const layout& layout, void* ptr);
    virtual memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) = 0;
    memory_ptr allocate_memory(const layout& layout, bool reset = true);

    virtual memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) = 0;
    virtual memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) = 0;
    memory_ptr share_buffer(const layout& layout, shared_handle buf);
    memory_ptr share_image(const layout& layout, shared_handle img);
    /// Create shared memory object on @p engine over specified @p plane of video decoder surface @p surf using specified @p layout
#ifdef _WIN32
    memory_ptr share_surface(const layout& layout, shared_handle surf, uint32_t plane);
    memory_ptr share_dx_buffer(const layout& layout, shared_handle res);
#else
    memory_ptr share_surface(const layout& layout, shared_surface surf, uint32_t plane);
#endif

    virtual bool is_the_same_buffer(const memory& mem1, const memory& mem2) = 0;

    virtual allocation_type get_default_allocation_type() const = 0;
    allocation_type get_lockable_preffered_memory_allocation_type(bool is_image_layout = false) const;
    bool supports_allocation(allocation_type type) const;

    const engine_configuration& configuration() const { return _configuration; }
    device_info get_device_info() const;
    const device::ptr get_device() const;
    memory_pool& get_memory_pool();
    virtual void* get_user_context() const = 0;

    uint64_t get_max_used_device_memory() const;
    uint64_t get_used_device_memory() const;

    bool use_unified_shared_memory() const;

    virtual stream_ptr create_stream() const = 0;
    virtual stream& get_program_stream() const = 0;

    static std::shared_ptr<cldnn::engine> create(engine_types engine_type,
                                                 runtime_types runtime_type,
                                                 const device::ptr device,
                                                 const engine_configuration& configuration = engine_configuration());

    static std::shared_ptr<cldnn::engine> create(engine_types engine_type,
                                                 runtime_types runtime_type,
                                                 const engine_configuration& configuration = engine_configuration());

protected:
    std::unique_ptr<memory_pool> _memory_pool;
    const device::ptr _device;
    engine_configuration _configuration;
};

std::shared_ptr<cldnn::engine> create_ocl_engine(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration);

}  // namespace cldnn
