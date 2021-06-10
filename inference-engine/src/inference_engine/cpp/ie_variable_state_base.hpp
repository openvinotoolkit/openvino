// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "cpp/exception2status.hpp"
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "ie_imemory_state.hpp"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief Default implementation for IVariableState
  * @ingroup ie_dev_api_variable_state_api
 */
class VariableStateBase : public IVariableState {
    std::shared_ptr<IVariableStateInternal> impl;

public:
    /**
     * @brief Constructor with actual underlying implementation.
     * @param impl Underlying implementation of type IVariableStateInternal
     */
    explicit VariableStateBase(std::shared_ptr<IVariableStateInternal> impl): impl(impl) {
        if (impl == nullptr) {
            IE_THROW() << "VariableStateBase implementation is not defined";
        }
    }

    StatusCode GetName(char* name, size_t len, ResponseDesc* resp) const noexcept override {
        for (size_t i = 0; i != len; i++) {
            name[i] = 0;
        }
        DescriptionBuffer buf(name, len);
        TO_STATUS(buf << impl->GetName());
        return OK;
    }

    StatusCode Reset(ResponseDesc* resp) noexcept override {
        TO_STATUS(impl->Reset());
    }

    StatusCode SetState(Blob::Ptr newState, ResponseDesc* resp) noexcept override {
        TO_STATUS(impl->SetState(newState));
    }

    StatusCode GetState(Blob::CPtr& state, ResponseDesc* resp) const noexcept override {
        TO_STATUS(state = impl->GetState());
    }
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
