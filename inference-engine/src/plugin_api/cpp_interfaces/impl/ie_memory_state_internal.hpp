// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_imemory_state_internal.hpp>
#include <string>

namespace InferenceEngine {

/**
 * @brief minimal interface for memory state implementation
 * @ingroup ie_dev_api_mem_state_api
 */
class VariableStateInternal : public IVariableStateInternal {
    std::string name;
    Blob::Ptr state;

public:
    explicit VariableStateInternal(std::string name): name(name) {}
    std::string GetName() const override {
        return name;
    }
    void SetState(Blob::Ptr newState) override {
        state = newState;
    }
    Blob::CPtr GetState() const override {
        return state;
    }
};

/*
 * @brief For compatibility reasons.
 */
using MemoryStateInternal = VariableStateInternal;
}  // namespace InferenceEngine
