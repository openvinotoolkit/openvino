// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <cpp_interfaces/interface/ie_imemory_state_internal.hpp>


namespace InferenceEngine {

/**
 * @brief minimal interface for memory state implementation
 */
class MemoryStateInternal : public IMemoryStateInternal {
    std::string name;
    Blob::Ptr state;

 public:
    explicit MemoryStateInternal(std::string name) : name(name) {
    }
    std::string GetName() const override {
        return name;
    }
    void SetState(Blob::Ptr newState) override {
        state = newState;
    }
    Blob::CPtr GetLastState() const override {
        return state;
    }
};


}  // namespace InferenceEngine