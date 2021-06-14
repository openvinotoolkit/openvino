// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <details/ie_so_pointer.hpp>

#include <memory>
#include <string>

namespace InferenceEngine {

/**
 * @interface IVariableStateInternal
 * @brief Minimal interface for variable state implementation
 * @ingroup ie_dev_api_variable_state_api
 */
class INFERENCE_ENGINE_API_CLASS(IVariableStateInternal) : public std::enable_shared_from_this<IVariableStateInternal> {
public:
    /**
     * @brief A shared pointer to a IVariableStateInternal interface
     */
    using Ptr = std::shared_ptr<IVariableStateInternal>;

    explicit IVariableStateInternal(const std::string& name);

    /**
     * @brief Gets a variable state name
     * @return A string representing variable state name
     */
    virtual std::string GetName() const;

    /**
     * @brief Reset internal variable state for relevant infer request, to a value specified as
     * default for according `ReadValue` node
     */
    virtual void Reset();

    /**
     * @brief Sets the new state for the next inference
     * @param newState A new state
     */
    virtual void SetState(const Blob::Ptr& newState);

    /**
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    virtual Blob::CPtr GetState() const;

    /**
     * @deprecated Use IVariableStateInternal::GetState method instead
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    INFERENCE_ENGINE_DEPRECATED("Use IVariableStateInternal::GetState method instead")
    virtual Blob::CPtr GetLastState() const;

protected:
    /**
     * @brief A default dtor
     */
    ~IVariableStateInternal() = default;

    std::string name;
    Blob::Ptr state;
};

/**
 * @brief For compatibility reasons.
 */
using IMemoryStateInternal = IVariableStateInternal;

/**
 * @brief SOPointer to IVariableStateInternal.
 */
using SoIVariableStateInternal = details::SOPointer<IVariableStateInternal>;

/**
 * @brief For compatibility reasons.
 */
using MemoryStateInternal = IVariableStateInternal;

}  // namespace InferenceEngine
