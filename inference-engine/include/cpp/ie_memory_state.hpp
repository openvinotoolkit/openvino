// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides VariableState
 *
 * @file ie_memory_state.hpp
 */

#pragma once

#include <string>
#include <memory>

#include "ie_api.h"
#include "ie_blob.h"
#include "details/ie_so_loader.h"

namespace InferenceEngine {

class IVariableStateInternal;

/**
 * @brief VariableState class
 */
class INFERENCE_ENGINE_API_CLASS(VariableState) {
    details::SharedObjectLoader              _so;
    std::shared_ptr<IVariableStateInternal>  _impl;

    /**
     * @brief Constructs VariableState from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Optional: Plugin to use. This is required to ensure that VariableState can work properly even if plugin object is destroyed.
     */
    VariableState(const details::SharedObjectLoader&             so,
                  const std::shared_ptr<IVariableStateInternal>& impl);
    friend class InferRequest;
    friend class ExecutableNetwork;

public:
    /**
     * @brief Default constructor
     */
    VariableState() = default;

    /**
     * @brief Reset internal variable state for relevant infer request,
     * to a value specified as default for according ReadValue node
     */
    void Reset();

    /**
     * @brief Gets name of current variable state, if length of array is not enough name is truncated by len, null
     * terminator is inserted as well. As variable state name `variable_id` from according `ReadValue` used. 
     * @return A string representing a state name
     */
    std::string GetName() const;

    /**
     * @brief Returns the value of the variable state.
     * @return A blob representing a state
     */
    Blob::CPtr GetState() const;

    /**
     * @brief Sets the new state for the next inference.
     * @param state The current state to set
     */
    void SetState(Blob::Ptr state);
};

/**
 * @brief For compatibility reasons.
 */
using MemoryState = VariableState;

}  // namespace InferenceEngine
