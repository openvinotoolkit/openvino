// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the IE RemoteContext and RemoteBlob classes
 *
 * @file ie_remote_context.hpp
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <map>
#include <memory>
#include <string>

#include "ie_api.h"
#include "ie_parameter.hpp"
#include "ie_remote_blob.hpp"

namespace InferenceEngine {
IE_SUPPRESS_DEPRECATED_START
/**
 * @brief This class represents an Inference Engine abstraction
 * for remote (non-CPU) accelerator device-specific execution context.
 * Such context represents a scope on the device within which executable
 * networks and remote memory blobs can exist, function and exchange data.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(RemoteContext)
    : public std::enable_shared_from_this<RemoteContext> {
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
        return dynamic_cast<T*>(GetHardwareContext().get()) != nullptr;
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
        return dynamic_cast<const T*>(GetHardwareContext().get()) != nullptr;
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
    T* as() noexcept {
        return dynamic_cast<T*>(GetHardwareContext().get());
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
    const T* as() const noexcept {
        return dynamic_cast<const T*>(GetHardwareContext().get());
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
     * @brief Allocates host accessible memory blob friendly for the device in current context
     * Returns a pointer to the object which implements MemoryBlob interface.
     * @param tensorDesc Defines the layout and dims of the blob
     * @return A pointer to host accessible MemoryBlob object
     */
    virtual MemoryBlob::Ptr CreateHostBlob(const TensorDesc& tensorDesc);

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

    /**
     * @brief Unwrap hardware remote context
     *
     * @return shared pointer to plugin specific remote context
     */
    const std::shared_ptr<InferenceEngine::RemoteContext> GetHardwareContext();

    /**
     * @brief Unwrap hardware remote context
     *
     * @return shared pointer to plugin specific remote context
     */
    const std::shared_ptr<const InferenceEngine::RemoteContext> GetHardwareContext() const;
};

/**
 * @brief A wrapper of CreateBlob method of RemoteContext to keep consistency with
 * plugin-specific wrappers.
 * @param desc Defines the layout and dims of the blob
 * @param ctx Pointer to the plugin object derived from RemoteContext.
 * @return A pointer to plugin object that implements RemoteBlob interface.
 */
inline INFERENCE_ENGINE_1_0_DEPRECATED RemoteBlob::Ptr make_shared_blob(const TensorDesc& desc,
                                                                        RemoteContext::Ptr ctx) {
    return ctx->CreateBlob(desc);
}

IE_SUPPRESS_DEPRECATED_END
}  // namespace InferenceEngine
