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

#include "ie_blob.h"
#include "ie_parameter.hpp"

namespace InferenceEngine {
class RemoteContext;

/**
 * @brief This class represents an Inference Engine abstraction to the memory allocated
 * on the remote (non-CPU) accelerator device
 */
class RemoteBlob : public MemoryBlob {
public:
    /**
     * @brief A smart pointer to the RemoteBlob object
     */
    using Ptr = std::shared_ptr<RemoteBlob>;

    /**
     * @brief A smart pointer to the const RemoteBlob object
     */
    using CPtr = std::shared_ptr<const RemoteBlob>;

    /**
     * @brief RemoteBlob virtual destructor
     */
    virtual ~RemoteBlob() = default;

    /**
     * @brief Constructor. Creates an empty RemoteBlob object with the specified precision.
     * @param tensorDesc Defines the layout and dims of the blob
     */
    explicit RemoteBlob(const TensorDesc& tensorDesc): MemoryBlob(tensorDesc) {}

    /**
    * @brief Returns the number of bytes per element.
    */
    size_t element_size() const noexcept override {
        return tensorDesc.getPrecision().size();
    }

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with underlying object.
     * Parameters include device/context/surface/buffer handles, access flags,
     * etc. Contents of the map returned depend on remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/parameter elements.
     */
    virtual ParamMap getParams() const = 0;

    /**
     * @brief Returns name of the device on which underlying object is allocated.
     * Abstract method.
     * @return A device name string in the same format as that in plugin metric.
     */
    virtual std::string getDeviceName() const noexcept = 0;

    /**
     * @brief Returns device context which underlying object belongs to.
     * Abstract method.
     * @return Pointer to plugin-specific context class object, which is derived from RemoteContext.
     * Dynamic casting should be used if it is necessary to retrieve a pointer to original class.
     */
    virtual std::shared_ptr<RemoteContext> getContext() const noexcept = 0;
};

/**
 * @brief This class represents an Inference Engine abstraction
 * for remote (non-CPU) accelerator device-specific execution context.
 * Such context represents a scope on the device within which executable
 * networks and remote memory blobs can exist, function and exchange data.
 */
class RemoteContext {
public:
    /**
     * @brief A smart pointer to the RemoteContext object
     */
    using Ptr = std::shared_ptr<RemoteContext>;

    /**
     * @brief A smart pointer to the const RemoteContext object
     */
    using CPtr = std::shared_ptr<const RemoteContext>;

    /**
     * @brief RemoteContext virtual destructor
     */
    virtual ~RemoteContext() = default;

    /**
     * @brief Checks if the RemoteContext object can be cast to the type T*
     *
     * @tparam T Type to be checked. Must represent a class derived from the RemoteContext
     * @return true if this object can be dynamically cast to the type T*. Otherwise, false
     */
    template <typename T,
        typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
        typename std::enable_if<std::is_base_of<RemoteContext, T>::value, int>::type = 0>
    bool is() noexcept {
        return dynamic_cast<T*>(this) != nullptr;
    }

    /**
     * @brief Checks if the RemoteContext object can be cast to the type const T*
     *
     * @tparam T Type to be checked. Must represent a class derived from the RemoteContext
     * @return true if this object can be dynamically cast to the type const T*. Otherwise, false
     */
    template <typename T,
        typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
        typename std::enable_if<std::is_base_of<RemoteContext, T>::value, int>::type = 0>
    bool is() const noexcept {
        return dynamic_cast<const T*>(this) != nullptr;
    }

    /**
     * @brief Casts this RemoteContext object to the type T*.
     *
     * @tparam T Type to cast to. Must represent a class derived from the RemoteContext
     * @return Raw pointer to the object of the type T or nullptr on error
     */
    template <typename T,
        typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
        typename std::enable_if<std::is_base_of<RemoteContext, T>::value, int>::type = 0>
    T * as() noexcept {
        return dynamic_cast<T*>(this);
    }

    /**
     * @brief Casts this RemoteContext object to the type const T*.
     *
     * @tparam T Type to cast to. Must represent a class derived from the RemoteContext
     * @return Raw pointer to the object of the type const T or nullptr on error
     */
    template <typename T,
        typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
        typename std::enable_if<std::is_base_of<RemoteContext, T>::value, int>::type = 0>
    const T * as() const noexcept {
        return dynamic_cast<const T*>(this);
    }

    /**
     * @brief Returns name of the device on which underlying object is allocated.
     * Abstract method.
     * @return A device name string in the same format as that in plugin metric.
     */
    virtual std::string getDeviceName() const noexcept = 0;

    /**
     * @brief Allocates memory blob in device memory or wraps user-supplied memory handle
     * using the specified tensor description and low-level device-specific parameters.
     * Returns a pointer to the object which implements RemoteBlob interface.
     * @param tensorDesc Defines the layout and dims of the blob
     * @param params Map of the low-level blob object parameters.
     * Abstract method.
     * @return A pointer to plugin object that implements RemoteBlob interface.
     */
    virtual RemoteBlob::Ptr CreateBlob(const TensorDesc& tensorDesc, const ParamMap& params = {}) = 0;

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with underlying object.
     * Parameters include device/context handles, access flags,
     * etc. Contents of the map returned depend on remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/parameter elements.
     */
    virtual ParamMap getParams() const = 0;
};

/**
 * @brief A wrapper of CreateBlob method of RemoteContext to keep consistency with
 * plugin-specific wrappers.
 * @param desc Defines the layout and dims of the blob
 * @param ctx Pointer to the plugin object derived from RemoteContext.
 * @return A pointer to plugin object that implements RemoteBlob interface.
 */
inline RemoteBlob::Ptr make_shared_blob(const TensorDesc& desc, RemoteContext::Ptr ctx) {
    return ctx->CreateBlob(desc);
}

}  // namespace InferenceEngine
