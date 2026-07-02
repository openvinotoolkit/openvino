// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ze_common.hpp"

#include <cstdint>
#include <memory>

namespace cldnn {
namespace ze {

/// @brief Defines supported Level Zero resources
enum class ze_resource_type : uint8_t {
    driver,
    device,
    context,
    command_queue,
    command_list,
    module,
    kernel,
    event_pool,
    event,
    image,
    fence,
    module_build_log,
    usm_memory,
};

/// @brief Provides information about specific Level Zero resource
template <ze_resource_type resource_type>
struct ze_resource_info {
    static_assert(false, "Specialization for given resource type is not implemented");
};

template <>
struct ze_resource_info<ze_resource_type::driver> {
    static constexpr ze_resource_type resource = ze_resource_type::driver;
    using handle_t = ze_driver_handle_t;
    // Driver is a read only global construct that is not released
    struct deleter_t {
        void operator()(handle_t handle) const {}
    };
};
template <>
struct ze_resource_info<ze_resource_type::device> {
    static constexpr ze_resource_type resource = ze_resource_type::device;
    using handle_t = ze_device_handle_t;
    // Device is a read only global construct that is not released
    struct deleter_t {
        void operator()(handle_t handle) const {}
    };
};
template <>
struct ze_resource_info<ze_resource_type::context> {
    static constexpr ze_resource_type resource = ze_resource_type::context;
    using handle_t = ze_context_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeContextDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::command_queue> {
    static constexpr ze_resource_type resource = ze_resource_type::command_queue;
    using handle_t = ze_command_queue_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeCommandQueueDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::command_list> {
    static constexpr ze_resource_type resource = ze_resource_type::command_list;
    using handle_t = ze_command_list_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeCommandListDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::module> {
    static constexpr ze_resource_type resource = ze_resource_type::module;
    using handle_t = ze_module_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeModuleDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::kernel> {
    static constexpr ze_resource_type resource = ze_resource_type::kernel;
    using handle_t = ze_kernel_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeKernelDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::event_pool> {
    static constexpr ze_resource_type resource = ze_resource_type::event_pool;
    using handle_t = ze_event_pool_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeEventPoolDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::event> {
    static constexpr ze_resource_type resource = ze_resource_type::event;
    using handle_t = ze_event_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeEventDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::image> {
    static constexpr ze_resource_type resource = ze_resource_type::image;
    using handle_t = ze_image_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeImageDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::fence> {
    static constexpr ze_resource_type resource = ze_resource_type::fence;
    using handle_t = ze_fence_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeFenceDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::module_build_log> {
    static constexpr ze_resource_type resource = ze_resource_type::module_build_log;
    using handle_t = ze_module_build_log_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeModuleBuildLogDestroy(handle));
        }
    };
};

struct ov_ze_usm_handle {
    ze_context_handle_t context;
    void* ptr;
};

template <>
struct ze_resource_info<ze_resource_type::usm_memory> {
    static constexpr ze_resource_type resource = ze_resource_type::usm_memory;
    using handle_t = ov_ze_usm_handle;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeMemFree(handle.context, handle.ptr));
        }
    };
};

/// @brief Generic RAII owner that releases resources when destroyed. All resources should have exactly one owner.
/// @tparam _resource_info_t Provides information on how to handle resource. Must define appropriate handle_t and deleter_t types.
template <typename _resource_info_t>
struct resource_owner {
public:
    using ptr = std::shared_ptr<resource_owner>;
    using handle_t = typename _resource_info_t::handle_t;
    using deleter_t = typename _resource_info_t::deleter_t;

    explicit resource_owner(handle_t handle, bool is_borrowed = false) : _handle(handle), _is_borrowed(is_borrowed) {}
    resource_owner(const resource_owner& other) = delete;
    resource_owner& operator=(const resource_owner& other) = delete;
    resource_owner(resource_owner&& other) {
        _handle = other._handle;
        _is_borrowed = other._is_borrowed;
        other._is_borrowed = true; // Mark as borrowed to prevent double free
    }
    resource_owner& operator=(resource_owner&& other) {
        if (this != &other) {
            release();
            _handle = other._handle;
            _is_borrowed = other._is_borrowed;
            other._is_borrowed = true; // Mark as borrowed to prevent double free
        }
        return *this;
    }
    ~resource_owner() {
        release();
    }

    handle_t get_handle() const {
        return _handle;
    }
private:
    void release() {
        if (!_is_borrowed) {
            deleter_t{}(_handle);
        }
    }

    handle_t _handle;
    bool _is_borrowed;
};


/// @brief Resource owner for Level Zero resources.
/// @tparam _resource_type Level Zero resource type managed by this owner.
template <ze_resource_type _resource_type>
using ze_owner = resource_owner<ze_resource_info<_resource_type>>;

}  // namespace ze
}  // namespace cldnn
