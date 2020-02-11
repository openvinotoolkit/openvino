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
    explicit MemoryState(IMemoryState::Ptr pState): actual(pState) {}

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
     */
    void SetState(Blob::Ptr state) {
        CALL_STATUS_FNC(SetState, state);
    }
};

}  // namespace InferenceEngine