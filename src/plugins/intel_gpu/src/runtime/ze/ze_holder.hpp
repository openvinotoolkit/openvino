// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "ze_common.hpp"

#include <cstdint>
#include <variant>
#include <memory>

namespace cldnn {
namespace ze {

enum class ze_resource_type : uint8_t {
    context,
    command_queue,
    command_list,
    module,
    kernel,
    event_pool,
    event,
    counter_based_event,
    image,
    fence,
    module_build_log,
    usm_memory,
};

template <ze_resource_type resource_type>
struct ze_resource_info {
    static_assert(false, "Specialization for given resource type is not implemented");
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::context;
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::context;
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::context;
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::module;
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::context;
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::event_pool;
    using handle_t = ze_event_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeEventDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::counter_based_event> {
    static constexpr ze_resource_type resource = ze_resource_type::counter_based_event;
    static constexpr ze_resource_type parent_resource = ze_resource_type::context;
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::context;
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::command_queue;
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
    static constexpr ze_resource_type parent_resource = ze_resource_type::context;
    using handle_t = ze_module_build_log_handle_t;
    struct deleter_t {
        void operator()(handle_t handle) const {
            OV_ZE_WARN(zeModuleBuildLogDestroy(handle));
        }
    };
};

template <>
struct ze_resource_info<ze_resource_type::usm_memory> {
    static constexpr ze_resource_type resource = ze_resource_type::usm_memory;
    static constexpr ze_resource_type parent_resource = ze_resource_type::context;
    using handle_t = void*;
    struct deleter_t {
        void operator()(ze_context_handle_t context, handle_t handle) const {
            OV_ZE_WARN(zeMemFree(context, handle));
        }
    };
};

template <ze_resource_type resource_type>
struct ze_resource {
public:
    using handle_t = typename ze_resource_info<resource_type>::handle_t;
    using deleter_t = typename ze_resource_info<resource_type>::deleter_t;

    explicit ze_resource(handle_t handle, bool take_ownership = true) : _handle(handle), _is_owner(take_ownership) {
        OPENVINO_ASSERT(_handle != nullptr, "[GPU] Can not create ze_resource with nullptr handle");
    }
    ze_resource(const ze_resource<resource_type>& other) = delete;
    ze_resource& operator=(const ze_resource<resource_type>& other) = delete;
    ~ze_resource() {
        if (_is_owner) {
            deleter_t{}(_handle);
        }
    }
    handle_t get_handle() const {
        return _handle;
    }
private:
    const handle_t _handle;
    const bool _is_owner;
};

template <>
struct ze_resource<ze_resource_type::usm_memory> {
public:
    static constexpr ze_resource_type resource_type = ze_resource_type::usm_memory;
    using handle_t = typename ze_resource_info<ze_resource_type::usm_memory>::handle_t;
    using deleter_t = typename ze_resource_info<ze_resource_type::usm_memory>::deleter_t;

    explicit ze_resource(ze_context_handle_t context, handle_t handle, bool take_ownership = true) : _context(context), _handle(handle), _is_owner(take_ownership) {
        OPENVINO_ASSERT(_context != nullptr, "[GPU] Can not create ze_resource with nullptr context");
        OPENVINO_ASSERT(_handle != nullptr, "[GPU] Can not create ze_resource with nullptr handle");
    }
    ze_resource(const ze_resource<resource_type>& other) = delete;
    ze_resource& operator=(const ze_resource<resource_type>& other) = delete;
    ~ze_resource() {
        if (_is_owner) {
            deleter_t{}(_context, _handle);
        }
    }
    handle_t get_handle() const {
        return _handle;
    }
private:
    const ze_context_handle_t _context;
    const handle_t _handle;
    const bool _is_owner;
};

template <ze_resource_type _resource_type>
struct ze_holder {
public:
    static constexpr ze_resource_type resource_type = _resource_type;
    static constexpr ze_resource_type parent_resource = ze_resource_info<resource_type>::parent_resource;
    using parent_holder_t = ze_holder<parent_resource>;
    using handle_t = typename ze_resource_info<resource_type>::handle_t;

    ze_holder() = default;
    ze_holder(const ze_holder &) = default;
    ze_holder(ze_holder&&) = default;
    ze_holder& operator=(const ze_holder&) = default;
    ze_holder& operator=(ze_holder&&) = default;
    explicit ze_holder(handle_t handle, parent_holder_t parent, bool take_ownership = true) {
        OPENVINO_ASSERT(!parent.is_empty(), "[GPU] Parent holder can not be empty when creating holder");
        // No need to check handle as resource ctor will throw for nullptr
        std::shared_ptr<ze_resource<resource_type>> resource;
        if constexpr (resource_type == ze_resource_type::usm_memory) {
            // USM memory requires context (parent) for destruction, so we need to pass it to the resource
            resource = std::make_shared<ze_resource<resource_type>>(parent.get_handle(), handle, take_ownership);
        } else {
            resource = std::make_shared<ze_resource<resource_type>>(handle, take_ownership);
        }
        _holder = std::move(resource);
        _parent = std::move(parent);
    }
    ~ze_holder() {
        drop();
    }

    handle_t get_handle() const {
        OPENVINO_ASSERT(_holder != nullptr, "[GPU] Attempt to get handle from an empty holder");
        return _holder->get_handle();
    }

    parent_holder_t get_parent() const {
        return _parent;
    }

    bool is_empty() const {
        return _holder == nullptr;
    }

    void drop() {
        // Release holder before parent to ensure correct destruction order
        _holder.reset();
        _parent.drop();
    }
private:
    std::shared_ptr<ze_resource<resource_type>> _holder;
    parent_holder_t _parent;
};

template <>
struct ze_holder<ze_resource_type::context> {
public:
    static constexpr ze_resource_type resource_type = ze_resource_type::context;
    using handle_t = typename ze_resource_info<resource_type>::handle_t;

    ze_holder() = default;
    ze_holder(const ze_holder &) = default;
    ze_holder(ze_holder&&) = default;
    ze_holder& operator=(const ze_holder&) = default;
    ze_holder& operator=(ze_holder&&) = default;
    explicit ze_holder(handle_t handle, bool take_ownership = true)
        : _holder(std::make_shared<ze_resource<resource_type>>(handle, take_ownership)) {}
    ~ze_holder() {
        drop();
    }

    handle_t get_handle() const {
        OPENVINO_ASSERT(_holder != nullptr, "[GPU] Attempt to get handle from an empty holder");
        return _holder->get_handle();
    }

    bool is_empty() const {
        return _holder == nullptr;
    }

    void drop() {
        _holder.reset();
    }
private:
    std::shared_ptr<ze_resource<resource_type>> _holder;
};

using ze_holder_variant = std::variant<
    std::monostate,
    ze_holder<ze_resource_type::context>,
    ze_holder<ze_resource_type::command_queue>,
    ze_holder<ze_resource_type::command_list>,
    ze_holder<ze_resource_type::module>,
    ze_holder<ze_resource_type::kernel>,
    ze_holder<ze_resource_type::event_pool>,
    ze_holder<ze_resource_type::event>,
    ze_holder<ze_resource_type::counter_based_event>,
    ze_holder<ze_resource_type::image>,
    ze_holder<ze_resource_type::fence>,
    ze_holder<ze_resource_type::module_build_log>,
    ze_holder<ze_resource_type::usm_memory>>;

}  // namespace ze
}  // namespace cldnn
