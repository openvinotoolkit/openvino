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
#include "details/ie_so_loader.h"
#include "ie_imemory_state.hpp"

namespace InferenceEngine {

class IVariableStateInternal;

/**
 * @brief C++ exception based error reporting wrapper of API class IVariableState
 */
class INFERENCE_ENGINE_API_CLASS(VariableState) {
    details::SharedObjectLoader              _so;
    std::shared_ptr<IVariableStateInternal>  _impl;
    IE_SUPPRESS_DEPRECATED_START
    std::shared_ptr<IVariableState>          actual;
    IE_SUPPRESS_DEPRECATED_END

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

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @deprecated This ctor will be removed in 2022.1
     * @brief constructs VariableState from the initialized std::shared_ptr
     * @param pState Initialized shared pointer
     * @param plg Optional: Plugin to use. This is required to ensure that VariableState can work properly even if plugin object is destroyed.
     */
    INFERENCE_ENGINE_DEPRECATED("This ctor will be removed in 2022.1")
    explicit VariableState(std::shared_ptr<IVariableState> pState,
                           std::shared_ptr<details::SharedObjectLoader> plg = {});
    IE_SUPPRESS_DEPRECATED_END

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
