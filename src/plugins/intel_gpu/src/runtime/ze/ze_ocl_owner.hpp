// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_owner.hpp"
#include "ze_ocl_common.hpp"

#include <cstdint>
#include <type_traits>
#include <tuple>
#include <optional>
#include <memory>

namespace cldnn {
namespace ze {

/// @brief Defines supported OpenCL resources for Level Zero interoperability
enum class ocl_resource_type : uint8_t {
    platform,
    device,
    context,
    command_queue,
    mem_object,
};

/// @brief Provides information about specific OpenCL resource
template <ocl_resource_type resource_type>
struct ocl_resource_info {
    static_assert(false, "Specialization for given resource type is not implemented");
};

template <>
struct ocl_resource_info<ocl_resource_type::platform> {
    static constexpr ocl_resource_type resource = ocl_resource_type::platform;
    using handle_t = cl_platform_id;
    // Platform resource is not managed
    struct deleter_t {
        void operator()(handle_t handle) const noexcept {}
    };
};

template <>
struct ocl_resource_info<ocl_resource_type::device> {
    static constexpr ocl_resource_type resource = ocl_resource_type::device;
    using handle_t = cl_device_id;
    // Device resource is not managed
    struct deleter_t {
        void operator()(handle_t handle) const noexcept {}
    };
};

template <>
struct ocl_resource_info<ocl_resource_type::context> {
    static constexpr ocl_resource_type resource = ocl_resource_type::context;
    using handle_t = cl_context;
    struct deleter_t {
        void operator()(handle_t handle) const noexcept {
            OV_OCL_WARN(clReleaseContext(handle));
        }
    };
};

template <>
struct ocl_resource_info<ocl_resource_type::command_queue> {
    static constexpr ocl_resource_type resource = ocl_resource_type::command_queue;
    using handle_t = cl_command_queue;
    struct deleter_t {
        void operator()(handle_t handle) const noexcept {
            OV_OCL_WARN(clReleaseCommandQueue(handle));
        }
    };
};

template <>
struct ocl_resource_info<ocl_resource_type::mem_object> {
    static constexpr ocl_resource_type resource = ocl_resource_type::mem_object;
    using handle_t = cl_mem;
    struct deleter_t {
        void operator()(handle_t handle) const noexcept {
            OV_OCL_WARN(clReleaseMemObject(handle));
        }
    };
};

/// @brief Resource owner for OpenCL resources.
/// @tparam _resource_type OpenCL resource type managed by this owner.

template <ocl_resource_type _resource_type>
using ocl_owner = resource_owner<ocl_resource_info<_resource_type>>;

/// @brief Resource owner for Level Zero resources and corresponding OpenCL resources.
///
/// OpenCL resources are optional and can be attached after owner is created.
/// When destroyed owner will release non-shared OpenCL resources in declaration order before destroying Level Zero resource.
/// @tparam _resource_type Level Zero resource type managed by this owner.
/// @tparam ...ocl_resource_types Types of OpenCL resources that can be attached.
template <ze_resource_type _resource_type, ocl_resource_type... ocl_resource_types>
struct ze_ocl_owner_impl {
    using ptr = std::shared_ptr<ze_ocl_owner_impl>;
    using ze_handle_t = typename ze_resource_info<_resource_type>::handle_t;
    ze_ocl_owner_impl(ze_handle_t ze_handle, bool is_borrowed = false) : _ze_owner(ze_handle, is_borrowed) {}

    ze_handle_t handle() const {
        return _ze_owner.get_handle();
    }

    template <ocl_resource_type ocl_resource_type>
    typename ocl_resource_info<ocl_resource_type>::handle_t ocl_handle() const {
        static_assert((((ocl_resource_type == ocl_resource_types) || ...)), "Specified OCL resource type can not be obtained from this owner");
        auto &owner = std::get<std::optional<ocl_owner<ocl_resource_type>>>(_ocl_owners);
        OPENVINO_ASSERT(owner.has_value(), "[GPU] Attempted to get ocl handle that is not attached to resource owner");
        return owner->get_handle();
    }

    template <ocl_resource_type ocl_resource_type>
    void attach_ocl_handle(typename ocl_resource_info<ocl_resource_type>::handle_t handle, bool is_borrowed = false) {
        static_assert((((ocl_resource_type == ocl_resource_types) || ...)), "Specified OCL resource type can not be attached to this owner");
        auto &owner = std::get<std::optional<ocl_owner<ocl_resource_type>>>(_ocl_owners);
        OPENVINO_ASSERT(!owner.has_value(), "[GPU] Attempted to overwrite existing ocl handle attached to resource owner");
        owner.emplace(handle, is_borrowed);
    }

    template <ocl_resource_type ocl_resource_type>
    void attach_ocl_handle(ocl_owner<ocl_resource_type> &&moved_owner) {
        static_assert((((ocl_resource_type == ocl_resource_types) || ...)), "Specified OCL resource type can not be attached to this owner");
        auto &owner = std::get<std::optional<ocl_owner<ocl_resource_type>>>(_ocl_owners);
        OPENVINO_ASSERT(!owner.has_value(), "[GPU] Attempted to overwrite existing ocl handle attached to resource owner");
        owner.emplace(std::move(moved_owner));
    }

    template <ocl_resource_type ocl_resource_type>
    bool has_ocl_handle() const {
        static_assert((((ocl_resource_type == ocl_resource_types) || ...)), "Specified OCL resource type can not be obtained from this owner");
        auto &owner = std::get<std::optional<ocl_owner<ocl_resource_type>>>(_ocl_owners);
        return owner.has_value();
    }

private:
    ze_owner<_resource_type> _ze_owner;
    std::tuple<std::optional<ocl_owner<ocl_resource_types>>...> _ocl_owners;
};

/// @brief Resource owner for Level Zero resources and corresponding OpenCL resources.
///
/// Only specific combinations of Level Zero and OpenCL resources are allowed, see specializations below.
/// @tparam _resource_type Level Zero resource type managed by this owner.
template <ze_resource_type _resource_type>
struct ze_ocl_owner : public ze_ocl_owner_impl<_resource_type> {
    using ze_ocl_owner_impl<_resource_type>::ze_ocl_owner_impl;
};

template <>
struct ze_ocl_owner<ze_resource_type::driver> : public ze_ocl_owner_impl<ze_resource_type::driver, ocl_resource_type::platform> {
    using ze_ocl_owner_impl<ze_resource_type::driver, ocl_resource_type::platform>::ze_ocl_owner_impl;
};
template <>
struct ze_ocl_owner<ze_resource_type::device> : public ze_ocl_owner_impl<ze_resource_type::device, ocl_resource_type::device> {
    using ze_ocl_owner_impl<ze_resource_type::device, ocl_resource_type::device>::ze_ocl_owner_impl;
};
template <>
struct ze_ocl_owner<ze_resource_type::context> : public ze_ocl_owner_impl<ze_resource_type::context, ocl_resource_type::context> {
    using ze_ocl_owner_impl<ze_resource_type::context, ocl_resource_type::context>::ze_ocl_owner_impl;
};
template <>
struct ze_ocl_owner<ze_resource_type::command_list> : public ze_ocl_owner_impl<ze_resource_type::command_list, ocl_resource_type::command_queue> {
    using ze_ocl_owner_impl<ze_resource_type::command_list, ocl_resource_type::command_queue>::ze_ocl_owner_impl;
};
template <>
struct ze_ocl_owner<ze_resource_type::usm_memory> : public ze_ocl_owner_impl<ze_resource_type::usm_memory, ocl_resource_type::mem_object> {
    using ze_ocl_owner_impl<ze_resource_type::usm_memory, ocl_resource_type::mem_object>::ze_ocl_owner_impl;
};
template <>
struct ze_ocl_owner<ze_resource_type::image> : public ze_ocl_owner_impl<ze_resource_type::image, ocl_resource_type::mem_object> {
    using ze_ocl_owner_impl<ze_resource_type::image, ocl_resource_type::mem_object>::ze_ocl_owner_impl;
};

}  // namespace ze
}  // namespace cldnn
