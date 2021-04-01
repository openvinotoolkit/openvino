// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

#include <memory>
#include <string>

namespace InferenceEngine {

/**
 * @interface IVariableStateInternal
 * @brief Minimal interface for variable state implementation
 * @ingroup ie_dev_api_variable_state_api
 */
class IVariableStateInternal {
public:
    /**
     * @brief A shared pointer to a IVariableStateInternal interface
     */
    using Ptr = std::shared_ptr<IVariableStateInternal>;

    /**
     * @brief A default virtual dtor
     */
    virtual ~IVariableStateInternal() = default;

    /**
     * @brief Gets a variable state name
     * @return A string representing variable state name
     */
    virtual std::string GetName() const = 0;

    /**
     * @brief Reset internal variable state for relevant infer request, to a value specified as
     * default for according `ReadValue` node
     */
    virtual void Reset() = 0;

    /**
     * @brief Sets the new state for the next inference
     * @param newState A new state
     */
    virtual void SetState(Blob::Ptr newState) = 0;

    /**
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    virtual Blob::CPtr GetState() const = 0;

    /**
     * @deprecated Use IVariableStateInternal::GetState method instead
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    INFERENCE_ENGINE_DEPRECATED("Use IVariableStateInternal::GetState method instead")
    virtual Blob::CPtr GetLastState() const {
        return GetState();
    }
};

/**
 * @brief For compatibility reasons.
 */
using IMemoryStateInternal = IVariableStateInternal;

}  // namespace InferenceEngine
