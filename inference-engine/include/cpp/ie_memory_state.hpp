// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

namespace InferenceEngine {

/**
 * @brief c++ exception based error reporting wrapper of API class IMemoryState
 */
class MemoryState {
    IMemoryState::Ptr actual = nullptr;

 public:
    /**
     * constructs MemoryState from the initialized shared_pointer
     * @param pState Initialized shared pointer
     */
    explicit MemoryState(IMemoryState::Ptr pState) : actual(pState) {}

    /**
     * @brief Wraps original method
     * IMemoryState::Reset
     */
    void Reset() {
        CALL_STATUS_FNC_NO_ARGS(Reset);
    }

    /**
     * @brief Wraps original method
     * IMemoryState::GetName
     */
    std::string GetName() const {
        char name[256];
        CALL_STATUS_FNC(GetName, name, sizeof(name));
        return name;
    }

    /**
     * @brief Wraps original method
     * IMemoryState::GetLastState
     */
    Blob::CPtr GetLastState() const {
        Blob::CPtr stateBlob;
        CALL_STATUS_FNC(GetLastState, stateBlob);
        return stateBlob;
    }

    /**
     * @brief Wraps original method
     * IMemoryState::SetState
     */
    void SetState(Blob::Ptr state) {
        CALL_STATUS_FNC(SetState, state);
    }
};

}  // namespace InferenceEngine