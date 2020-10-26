// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file
 */

#pragma once

#include <string>

#include "ie_imemory_state.hpp"

namespace InferenceEngine {

/**
 * @brief C++ exception based error reporting wrapper of API class IVariableState
 */
class VariableState {
    IVariableState::Ptr actual = nullptr;

public:
    /**
     * constructs VariableState from the initialized shared_pointer
     * @param pState Initialized shared pointer
     */
    explicit VariableState(IVariableState::Ptr pState): actual(pState) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "VariableState wrapper was not initialized.";
        }
    }

    /**
     * @copybrief IVariableState::Reset
     *
     * Wraps IVariableState::Reset
     */
    void Reset() {
        CALL_STATUS_FNC_NO_ARGS(Reset);
    }

    /**
     * @copybrief IVariableState::GetName
     *
     * Wraps IVariableState::GetName
     * @return A string representing a state name
     */
    std::string GetName() const {
        char name[256];
        CALL_STATUS_FNC(GetName, name, sizeof(name));
        return name;
    }

    /**
     * @copybrief IVariableState::GetState
     *
     * Wraps IVariableState::GetState
     * @return A blob representing a last state 
     */
    Blob::CPtr GetState() const {
        Blob::CPtr stateBlob;
        CALL_STATUS_FNC(GetState, stateBlob);
        return stateBlob;
    }

    INFERENCE_ENGINE_DEPRECATED("Use GetState function instead")
    Blob::CPtr GetLastState() const {
        return GetState();
    }

    /**
     * @copybrief IVariableState::SetState
     *
     * Wraps IVariableState::SetState
     * @param state The current state to set
     */
    void SetState(Blob::Ptr state) {
        CALL_STATUS_FNC(SetState, state);
    }
};

/*
 * @brief For compatibility reasons.
 */
using MemoryState = VariableState;
}  // namespace InferenceEngine
