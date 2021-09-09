// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the IE RemoteContext and RemoteBlob classes
 *
 * @file ie_remote_context.hpp
 */
#pragma once

#include <memory>
#include <string>

#include "ie_parameter.hpp"
#include "ie_remote_context.hpp"

namespace InferenceEngine {

class RemoteBlob;

class INFERENCE_ENGINE_API_CLASS(IRemoteContext)
    : public RemoteContext,
      public std::enable_shared_from_this<RemoteContext> {
public:
    /**
     * @brief A smart pointer to the IRemoteContext object
     */
    using Ptr = std::shared_ptr<IRemoteContext>;

    /**
     * @brief Returns name of the device on which underlying object is allocated.
     * Abstract method.
     * @return A device name string in the same format as that in plugin metric.
     */
    std::string getDeviceName() const noexcept override;

    /**
     * @brief Allocates memory blob in device memory or wraps user-supplied memory handle
     * using the specified tensor description and low-level device-specific parameters.
     * Returns a pointer to the object which implements RemoteBlob interface.
     * @param tensorDesc Defines the layout and dims of the blob
     * @param params Map of the low-level blob object parameters.
     * Abstract method.
     * @return A pointer to plugin object that implements RemoteBlob interface.
     */
    std::shared_ptr<RemoteBlob> CreateBlob(const TensorDesc& tensorDesc, const ParamMap& params = {}) override;

    /**
     * @brief Returns a map of device-specific parameters required for low-level
     * operations with underlying object.
     * Parameters include device/context handles, access flags,
     * etc. Contents of the map returned depend on remote execution context that is
     * currently set on the device (working scenario).
     * Abstract method.
     * @return A map of name/parameter elements.
     */
    ParamMap getParams() const override;

protected:
    /**
     * @brief IRemoteContext destructor
     */
    ~IRemoteContext() = default;
};

}  // namespace InferenceEngine
