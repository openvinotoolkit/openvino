// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include <string>

namespace InferenceEngine {

/**
 * @brief Minimal interface for variable state implementation
 * @ingroup ie_dev_api_variable_state_api
 */
class VariableStateInternal : public IVariableStateInternal {
    std::string name;
    Blob::Ptr state;

public:
    /**
     * @brief Constructs a variable state with a given name
     * @param name A name of variable state
     */
    explicit VariableStateInternal(std::string name) : name(name) {}

    /**
     * @brief Gets a variable state name
     * @return A string representing variable state name
     */
    std::string GetName() const override {
        return name;
    }

    /**
     * @brief Sets the new state for the next inference
     * @param newState A new state
     */
    void SetState(Blob::Ptr newState) override {
        state = newState;
    }

    /**
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    Blob::CPtr GetState() const override {
        return state;
    }
};

/**
 * @brief For compatibility reasons.
 */
using MemoryStateInternal = VariableStateInternal;

}  // namespace InferenceEngine
