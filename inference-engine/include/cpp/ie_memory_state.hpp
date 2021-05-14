// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper classes for IVariableState
 *
 * @file ie_memory_state.hpp
 */

#pragma once

#include <string>
#include <memory>

#include "ie_api.h"
#include "ie_blob.h"
#include "details/ie_so_pointer.hpp"

namespace InferenceEngine {

namespace details {
class SharedObjectLoader;
}

class IVariableStateInternal;

/**
 * @brief C++ exception based error reporting wrapper of API class IVariableState
 */
class INFERENCE_ENGINE_API_CLASS(VariableState) : protected details::SOPointer<IVariableStateInternal> {
    using details::SOPointer<IVariableStateInternal>::SOPointer;
    friend class InferRequest;
    friend class ExecutableNetwork;

public:
    /**
     * @copybrief IVariableState::Reset
     *
     * Wraps IVariableState::Reset
     */
    void Reset();

    /**
     * @copybrief IVariableState::GetName
     *
     * Wraps IVariableState::GetName
     * @return A string representing a state name
     */
    std::string GetName() const;

    /**
     * @copybrief IVariableState::GetState
     *
     * Wraps IVariableState::GetState
     * @return A blob representing a state
     */
    Blob::CPtr GetState() const;

    /**
     * @copybrief IVariableState::GetLastState
     * @deprecated Use IVariableState::SetState instead
     *
     * Wraps IVariableState::GetLastState
     * @return A blob representing a last state 
     */
    INFERENCE_ENGINE_DEPRECATED("Use VariableState::GetState function instead")
    Blob::CPtr GetLastState() const;

    /**
     * @copybrief IVariableState::SetState
     *
     * Wraps IVariableState::SetState
     * @param state The current state to set
     */
    void SetState(Blob::Ptr state);
};

/**
 * @brief For compatibility reasons.
 */
using MemoryState = VariableState;

}  // namespace InferenceEngine
