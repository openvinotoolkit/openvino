// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the IE RemoteContext and RemoteBlob classes
 *
 * @file ie_remote_context.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "common.hpp"
#include "details/ie_so_loader.h"
#include "ie_parameter.hpp"
#include "ie_remote_context.hpp"

namespace InferenceEngine {
class IRemoteContext;
class RemoteBlob;
}  // namespace InferenceEngine

namespace ov {
namespace runtime {

class Core;

/**
 * @brief This class represents an abstraction
 * for remote (non-CPU) accelerator device-specific execution context.
 * Such context represents a scope on the device within which executable
 * networks and remote memory blobs can exist, function and exchange data.
 */
class INFERENCE_ENGINE_API_CLASS(RemoteContext) {
    ie::details::SharedObjectLoader _so;
    std::shared_ptr<ie::IRemoteContext> _impl;

    /**
     * @brief Constructs RemoteContext from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that RemoteContext can work properly even if plugin
     * object is destroyed.
     * @param impl Initialized shared pointer
     */
    RemoteContext(const ie::details::SharedObjectLoader& so, const std::shared_ptr<ie::IRemoteContext>& impl);
    friend class Core;

public:
    /**
     * @brief Default constructor
     */
    RemoteContext() = default;

    /**
     * @brief Checks if the RemoteContext object can be cast to the type T*
     *
     * @tparam T Type to be checked. Must represent a class derived from the RemoteContext
     * @return true if this object can be dynamically cast to the type T*. Otherwise, false
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<ie::RemoteContext, T>::value, int>::type = 0>
    bool is() noexcept {
        return dynamic_cast<T*>(_impl.get()) != nullptr;
    }

    /**
     * @brief Checks if the RemoteContext object can be cast to the type const T*
     *
     * @tparam T Type to be checked. Must represent a class derived from the RemoteContext
     * @return true if this object can be dynamically cast to the type const T*. Otherwise, false
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<ie::RemoteContext, T>::value, int>::type = 0>
    bool is() const noexcept {
        return dynamic_cast<const T*>(_impl.get()) != nullptr;
    }

    /**
     * @brief Casts this RemoteContext object to the type T*.
     *
     * @tparam T Type to cast to. Must represent a class derived from the RemoteContext
     * @return Raw pointer to the object of the type T or nullptr on error
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<ie::RemoteContext, T>::value, int>::type = 0>
    T* as() noexcept {
        return dynamic_cast<T*>(_impl.get());
    }

    /**
     * @brief Casts this RemoteContext object to the type const T*.
     *
     * @tparam T Type to cast to. Must represent a class derived from the RemoteContext
     * @return Raw pointer to the object of the type const T or nullptr on error
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<ie::RemoteContext, T>::value, int>::type = 0>
    const T* as() const noexcept {
        return dynamic_cast<const T*>(_impl.get());
    }

    /**
     * @brief Returns name of the device on which underlying object is allocated.
     * Abstract method.
     * @return A device name string in the same format as that in plugin metric.
     */
    std::string get_device_name() const;

    /**
     * @brief Allocates memory blob in device memory or wraps user-supplied memory handle
     * using the specified tensor description and low-level device-specific parameters.
     * Returns a pointer to the object which implements RemoteBlob interface.
     * @param tensorDesc Defines the layout and dims of the blob
     * @param params Map of the low-level blob object parameters.
     * Abstract method.
     * @return A pointer to plugin object that implements RemoteBlob interface.
     */
    std::shared_ptr<ie::RemoteBlob> create_blob(const ie::TensorDesc& tensorDesc, const ie::ParamMap& params = {});

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with underlying object.
     * Parameters include device/context handles, access flags,
     * etc. Contents of the map returned depend on remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/parameter elements.
     */
    ie::ParamMap get_params() const;
};

}  // namespace runtime
}  // namespace ov
