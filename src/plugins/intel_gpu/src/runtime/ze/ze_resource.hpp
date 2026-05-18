// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_ocl_owner.hpp"

namespace cldnn {
namespace ze {

/// @brief Level Zero resource that can optionally have attached OpenCL resources.
///
/// Copies of this class objects share the same resources. Resource are released when the last copy is destroyed.
/// @tparam _resource_type Level Zero resource type managed by this class.
template <ze_resource_type _resource_type>
class ze_resource {
public:
    using res = ze_resource_type;
    static constexpr ze_resource_type resource_type = _resource_type;
    using ze_handle_t = typename ze_resource_info<resource_type>::handle_t;
    using ze_ocl_owner_t = ze_ocl_owner<resource_type>;

    ze_resource() = default;
    ze_resource(const ze_resource &) = default;
    ze_resource(ze_resource&&) = default;
    ze_resource& operator=(const ze_resource&) = default;
    ze_resource& operator=(ze_resource&&) = default;

    /// @brief Create resource from existing holder.
    /// @param holder Resource holder.
    explicit ze_resource(typename ze_ocl_owner_t::ptr holder)
        : _holder(holder) {}

    /// @brief Create ze_resource from Level Zero handle. Assumes passed handle is valid.
    /// @param ze_handle Valid Level Zero object handle.
    /// @param is_shared if false, takes ownership of the handle.
    explicit ze_resource(ze_handle_t ze_handle, bool is_shared = false)
        : _holder(std::make_shared<ze_ocl_owner_t>(ze_handle, is_shared)) {}

    /// @brief Get Level Zero handle or throw if resource is empty.
    ze_handle_t get_ze_handle() const {
        OPENVINO_ASSERT(_holder != nullptr, "[GPU] Attempted to get Level Zero handle from empty resource");
        return _holder->get_ze_handle();
    }

    /// @brief Get OpenCL handle or throw if resource is empty.
    template <ocl_resource_type ocl_resource_type>
    typename ocl_resource_info<ocl_resource_type>::handle_t get_ocl_handle() const {
        OPENVINO_ASSERT(_holder != nullptr, "[GPU] Attempted to get OpenCL handle from empty resource");
        return _holder->template get_ocl_handle<ocl_resource_type>();
    }

    /// @brief Attach OpenCL handle to the resource or throw if resource is empty. Assumes passed handle is valid.
    ///
    /// This function won't release passed handle in case exception is thrown.
    template <ocl_resource_type ocl_resource_type>
    void attach_ocl_handle(typename ocl_resource_info<ocl_resource_type>::handle_t handle, bool is_shared = false) {
        OPENVINO_ASSERT(_holder != nullptr, "[GPU] Attempted to attach OpenCL handle to empty resource");
        return _holder->template attach_ocl_handle<ocl_resource_type>(handle, is_shared);
    }

    /// @brief Attach OpenCL handle to the resource or throw if resource is empty. Assumes passed handle is valid.
    ///
    /// This function will release passed handle in case exception is thrown.
    template <ocl_resource_type ocl_resource_type>
    void attach_ocl_handle(ocl_owner<ocl_resource_type> &&owner) {
        OPENVINO_ASSERT(_holder != nullptr, "[GPU] Attempted to attach OpenCL handle to empty resource");
        return _holder->template attach_ocl_handle<ocl_resource_type>(std::move(owner));
    }

    /// @brief Check if resource has specific OpenCL handle.
    template <ocl_resource_type ocl_resource_type>
    bool has_ocl_handle() const {
        if (_holder == nullptr) {
            return false;
        }
        return _holder->template has_ocl_handle<ocl_resource_type>();
    }

    /// @brief Get resource holder. Note that holder might be nullptr
    typename ze_ocl_owner_t::ptr get_holder() const {
        return _holder;
    }

    /// @brief Drop resources and reset to empty state
    void drop() {
        _holder.reset();
    }

    /// @brief Returns true if object is not managing any Level Zero resource, false otherwise.
    bool is_empty() const {
        return _holder == nullptr;
    }

private:
    typename ze_ocl_owner_t::ptr _holder;
};

using ze_driver_resource = ze_resource<ze_resource_type::driver>;
using ze_device_resource = ze_resource<ze_resource_type::device>;
using ze_context_resource = ze_resource<ze_resource_type::context>;
using ze_command_queue_resource = ze_resource<ze_resource_type::command_queue>;
using ze_command_list_resource = ze_resource<ze_resource_type::command_list>;
using ze_module_resource = ze_resource<ze_resource_type::module>;
using ze_kernel_resource = ze_resource<ze_resource_type::kernel>;
using ze_event_pool_resource = ze_resource<ze_resource_type::event_pool>;
using ze_event_resource = ze_resource<ze_resource_type::event>;
using ze_image_resource = ze_resource<ze_resource_type::image>;
using ze_fence_resource = ze_resource<ze_resource_type::fence>;
using ze_module_build_log_resource = ze_resource<ze_resource_type::module_build_log>;
using ze_usm_resource = ze_resource<ze_resource_type::usm_memory>;

}  // namespace ze
}  // namespace cldnn
