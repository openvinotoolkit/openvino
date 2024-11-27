// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory_caps.hpp"
#include "memory_pool.hpp"
#include "layout.hpp"
#include "execution_config.hpp"
#include "engine_configuration.hpp"

#include <memory>
#include <set>
#include <utility>
#include <string>
#include <atomic>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace cldnn {

class stream;

using memory_ptr = std::shared_ptr<memory>;
using stream_ptr = std::shared_ptr<stream>;

using primitive_id = std::string;

class engine {
public:
    /// Default destructor
    virtual ~engine() = default;

    /// Returns type of the engine
    virtual engine_types type() const = 0;
    /// Returns runtime type used in the engine
    virtual runtime_types runtime_type() const = 0;

    /// Create memory object attached to the buffer allocated by user.
    /// @param ptr  The pointer to user allocated buffer.
    /// @note Size (in bytes) of the buffer should be equal to @p layout.bytes_count()
    /// User is responsible for buffer deallocation. Buffer lifetime should be bigger than lifetime of the memory object.
    memory_ptr attach_memory(const layout& layout, void* ptr);

    /// Allocate gpu memory using specified @p layout and alloation @p type
    virtual memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) = 0;

    /// Allocate gpu memory using specified @p layout. Allocation type is selected automatically based on engine/device configuration
    memory_ptr allocate_memory(const layout& layout, bool reset = true);

    /// Created memory object from memory @p params and reinterpred the data using specified @p layout
    virtual memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) = 0;

    /// Created memory object from the other @p memory and reinterpred the data using specified @p new_layout
    virtual memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) = 0;

    /// Create shared memory object using user-supplied memory buffer @p buf using specified @p layout
    memory_ptr share_buffer(const layout& layout, shared_handle buf);

    /// Create shared memory object using user-supplied USM pointer @p usm_ptr using specified @p layout
    memory_ptr share_usm(const layout& layout, shared_handle usm_ptr);

    /// Create shared memory object using user-supplied 2D image @p img using specified @p layout
    memory_ptr share_image(const layout& layout, shared_handle img);

    /// Create shared memory object over specified @p plane of video decoder surface @p surf using specified @p layout
#ifdef _WIN32
    memory_ptr share_surface(const layout& layout, shared_handle surf, uint32_t plane);
    memory_ptr share_dx_buffer(const layout& layout, shared_handle res);
#else
    memory_ptr share_surface(const layout& layout, shared_surface surf, uint32_t plane);
#endif

    /// Checks whether two memory objects represents the same physical memory
    virtual bool is_the_same_buffer(const memory& mem1, const memory& mem2) = 0;

    virtual bool check_allocatable(const layout& layout, allocation_type type) = 0;

    /// Returns basic allocation type which will be used as a fallback when allocation type is not specified or device doesn't support some features.
    virtual allocation_type get_default_allocation_type() const = 0;

    /// Returns preferred allocation type which can be mapped to host ptr
    allocation_type get_lockable_preferred_memory_allocation_type(bool is_image_layout = false) const;

    /// Returns preferred device allocation type which may be not lockable
    allocation_type get_preferred_memory_allocation_type(bool is_image_layout = false) const;

    /// Checks if the current engine supports speicied allocation @p type
    bool supports_allocation(allocation_type type) const;

    /// Returns device structure which represents stores device capabilities
    const device_info& get_device_info() const;

    /// Returns device object associated with the engine
    const device::ptr get_device() const;

    /// Returns user context handle which was used to create the engine
    virtual void* get_user_context() const = 0;

    /// Returns the total maximum amount of GPU memory allocated by engine in current process for all allocation types
    uint64_t get_max_used_device_memory() const;

    /// Returns the maximum amount of GPU memory allocated by engine in current process for the specified allocation @p type
    uint64_t get_max_used_device_memory(allocation_type type) const;

    /// Returns the amount of GPU memory specified allocation @p type that currently used by the engine
    uint64_t get_used_device_memory(allocation_type type) const;

    /// Returns statistics of GPU memory allocated by engine in current process for all allocation types.
    /// @note It contains information about current memory usage
    std::map<std::string, uint64_t> get_memory_statistics() const;

    /// Adds @p bytes count to currently used memory size of the specified allocation @p type
    void add_memory_used(uint64_t bytes, allocation_type type);

    /// Subtracts @p bytes count from currently used memory size of the specified allocation @p type
    void subtract_memory_used(uint64_t bytes, allocation_type type);

    /// Returns true if USM is enabled in engine config and device/driver supports required features
    bool use_unified_shared_memory() const;

    /// Returns the size of the larger of the GPU memory and CPU memory.
    uint64_t get_max_memory_size() const;

    /// Returns the size of CPU memory.
    uint64_t get_host_memory_size() const;

    /// Create stream object for current engine
    virtual stream_ptr create_stream(const ExecutionConfig& config) const = 0;

    /// Creates stream object from user handle
    virtual stream_ptr create_stream(const ExecutionConfig& config, void *handle) const = 0;

    /// Returns service stream which can be used during program build and optimizations
    virtual stream& get_service_stream() const = 0;

    virtual allocation_type detect_usm_allocation_type(const void* memory) const = 0;

#ifdef ENABLE_ONEDNN_FOR_GPU
    /// Creates onednn engine object which shares device and context with current engine
    virtual void create_onednn_engine(const ExecutionConfig& config) = 0;

    /// Returns onednn engine object which shares device and context with current engine
    virtual dnnl::engine& get_onednn_engine() const = 0;
#endif

    /// This method is intended to create kernel handle for current engine from handle from arbitrary engine
    /// For instance, source kernel can be compiled using ocl engine, and then we can build L0 kernel object based on that
    virtual kernel::ptr prepare_kernel(const kernel::ptr kernel) const = 0;

    /// Factory method which creates engine object with impl configured by @p engine_type
    /// @param engine_type requested engine type
    /// @param runtime_type requested execution runtime for the engine. @note some runtime/engine types configurations might be unsupported
    /// @param device specifies the device which the engine is created for
    /// @param configuration options for the engine
    static std::shared_ptr<cldnn::engine> create(engine_types engine_type, runtime_types runtime_type, const device::ptr device);

    /// Factory method which creates engine object with impl configured by @p engine_type
    /// @param engine_type requested engine type
    /// @param runtime_type requested execution runtime for the engine. @note some runtime/engine types configurations might be unsupported
    /// @param configuration options for the engine
    /// @note engine is created for the first device returned by devices query
    static std::shared_ptr<cldnn::engine> create(engine_types engine_type, runtime_types runtime_type);

protected:
    /// Create engine for given @p device and @p configuration
    engine(const device::ptr device);
    const device::ptr _device;

    std::array<std::atomic<uint64_t>, static_cast<size_t>(allocation_type::max_value)> _memory_usage_data{};
    std::array<std::atomic<uint64_t>, static_cast<size_t>(allocation_type::max_value)> _peak_memory_usage_data{};
};

}  // namespace cldnn
