// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file
 */

#pragma once
#include <string>

namespace InferenceEngine {

/**
 * @brief C++ exception based error reporting wrapper of API class IMemoryState
 */
class MemoryState {
    IMemoryState::Ptr actual = nullptr;

public:
    /**
     * constructs MemoryState from the initialized shared_pointer
     * @param pState Initialized shared pointer
     */
    explicit MemoryState(IMemoryState::Ptr pState): actual(pState) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "MemoryState wrapper was not initialized.";
        }
    }

    /**
     * @copybrief IMemoryState::Reset
     *
     * Wraps IMemoryState::Reset
     */
    void Reset() {
        CALL_STATUS_FNC_NO_ARGS(Reset);
    }

    /**
     * @copybrief IMemoryState::GetName
     *
     * Wraps IMemoryState::GetName
     * @return A string representing a state name
     */
    std::string GetName() const {
        char name[256];
        CALL_STATUS_FNC(GetName, name, sizeof(name));
        return name;
    }

    /**
     * @copybrief IMemoryState::GetLastState
     *
     * Wraps IMemoryState::GetLastState
     * @return A blob representing a last state 
     */
    Blob::CPtr GetLastState() const {
        Blob::CPtr stateBlob;
        CALL_STATUS_FNC(GetLastState, stateBlob);
        return stateBlob;
    }

    /**
     * @copybrief IMemoryState::SetState
     *
     * Wraps IMemoryState::SetState
     * @param state The current state to set
     */
    void SetState(Blob::Ptr state) {
        CALL_STATUS_FNC(SetState, state);
    }
};

}  // namespace InferenceEngine