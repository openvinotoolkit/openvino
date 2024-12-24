// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace intel_cpu {

class RemoteContextImpl : public ov::IRemoteContext {
public:
    RemoteContextImpl(const std::string& device_name);

    /**
     * @brief Returns name of a device on which underlying object is allocated.
     * @return A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]` (e.g. GPU.0.1).
     */
    const std::string& get_device_name() const override;

    /**
     * @brief Returns a map of device-specific parameters
     * @return A map of name/Any elements.
     */
    const ov::AnyMap& get_property() const override;

    /**
     * @brief Allocates memory tensor in device memory or wraps user-supplied memory handle
     * using the specified tensor description and low-level device-specific parameters.
     * Returns a pointer to the object that implements the RemoteTensor interface.
     * @param type Defines the element type of the tensor.
     * @param shape Defines the shape of the tensor.
     * @param params Map of the low-level tensor object parameters.
     * @return Pointer to a plugin object that implements the RemoteTensor interface.
     */
    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params) override;

private:
    std::string m_device_name;
    ov::AnyMap properties;
};

}  // namespace intel_cpu
}  // namespace ov
