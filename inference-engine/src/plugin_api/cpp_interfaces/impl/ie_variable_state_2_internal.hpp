// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_variable_state_internal.hpp>
#include "cpp/ie_memory_state.hpp"

namespace InferenceEngine {

class VariableState2Internal : public VariableStateInternal {
    VariableState actual;

public:
    explicit VariableState2Internal(const VariableState & variableState) :
        VariableStateInternal(variableState.GetName()), actual(variableState) {
        // TODO: added a check for emptyness
        // if (!actual) {
        //     IE_THROW(NotAllocated);
        // }
    }

    void Reset() override {
        actual.Reset();
    }

    void SetState(Blob::Ptr newState) override {
        actual.SetState(newState);
    }

    Blob::CPtr GetState() const override {
        return actual.GetState();
    }
};

}  // namespace InferenceEngine
