// Copyright (C) 2018-2022 Intel Corporation
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
    explicit RemoteBlob(const TensorDesc& tensorDesc) : MemoryBlob(tensorDesc) {}

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
}  // namespace InferenceEngine
