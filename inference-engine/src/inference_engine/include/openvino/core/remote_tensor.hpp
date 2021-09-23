// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Runtime tensor API
 *
 * @file openvino/core/remote_tensor.hpp
 */
#pragma once

#include "ie_parameter.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/core/tensor.hpp"

namespace ov {
namespace runtime {
class RemoteContext;
}  // namespace runtime

/**
 * @brief Remote memory access and interpretation API
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class INFERENCE_ENGINE_API_CLASS(RemoteTensor) : public Tensor {
    using Tensor::Tensor;
    friend class ov::runtime::RemoteContext;

public:
    using Tensor::get_element_type;

    using Tensor::get_shape;

    void* data(const element::Type) = delete;

    template <typename T>
    T* data() = delete;

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with underlying object.
     * Parameters include device/context/surface/buffer handles, access flags,
     * etc. Contents of the map returned depend on remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/parameter elements.
     */
    ie::ParamMap get_params() const;

    /**
     * @brief Returns name of the device on which underlying object is allocated.
     * Abstract method.
     * @return A device name string in the same format as that in plugin metric.
     */
    std::string get_device_name() const;

    /**
     * @brief Checks if the RemoteContext object can be cast to the type T*
     *
     * @tparam T Type to be checked. Must represent a class derived from the RemoteContext
     * @return true if this object can be dynamically cast to the type T*. Otherwise, false
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<ie::RemoteBlob, T>::value, int>::type = 0>
    bool is() noexcept {
        return dynamic_cast<T*>(_impl.get()) != nullptr;
    }

    /**
     * @brief Checks if the RemoteContext object can be cast to the type const T*
     *
     * @tparam T Type to be checked. Must represent a class derived from the RemoteContext
     * @return true if this object can be dynamically cast to the type const T*. Otherwise, false
     */
    template <typename T>
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
              typename std::enable_if<std::is_base_of<ie::RemoteBlob, T>::value, int>::type = 0>
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
              typename std::enable_if<std::is_base_of<ie::RemoteBlob, T>::value, int>::type = 0>
    const T* as() const noexcept {
        return dynamic_cast<const T*>(_impl.get());
    }
};
}  // namespace ov