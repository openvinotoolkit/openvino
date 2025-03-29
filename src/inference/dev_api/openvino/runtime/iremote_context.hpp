// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime Remote Context interface
 * @file openvino/runtime/iremote_context.hpp
 */

#pragma once

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {

class OPENVINO_RUNTIME_API IRemoteContext : public std::enable_shared_from_this<IRemoteContext> {
public:
    virtual ~IRemoteContext() = default;

    /**
     * @brief Returns name of a device on which underlying object is allocated.
     * Abstract method.
     * @return A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]` (e.g. GPU.0.1).
     */
    virtual const std::string& get_device_name() const = 0;

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with underlying object.
     * Parameters include device/context handles, access flags,
     * etc. Contents of the map returned depend on remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/Any elements.
     */
    virtual const ov::AnyMap& get_property() const = 0;

    /**
     * @brief Allocates memory tensor in device memory or wraps user-supplied memory handle
     * using the specified tensor description and low-level device-specific parameters.
     * Returns a pointer to the object that implements the RemoteTensor interface.
     * @param type Defines the element type of the tensor.
     * @param shape Defines the shape of the tensor.
     * @param params Map of the low-level tensor object parameters.
     * @return Pointer to a plugin object that implements the RemoteTensor interface.
     */
    virtual ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                                       const ov::Shape& shape,
                                                       const ov::AnyMap& params = {}) = 0;

    /**
     * @brief This method is used to create a host tensor object friendly for the device in current context.
     * For example, GPU context may allocate USM host memory (if corresponding extension is available),
     * which could be more efficient than regular host memory.
     * @param type Tensor element type.
     * @param shape Tensor shape.
     * @return A tensor instance with device friendly memory.
     */
    virtual ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape);
};

}  // namespace ov
