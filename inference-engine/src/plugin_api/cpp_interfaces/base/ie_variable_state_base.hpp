// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/impl/ie_variable_state_internal.hpp"
#include "ie_imemory_state.hpp"

namespace InferenceEngine {

/**
 * @brief Default implementation for IVariableState
 * @tparam T Minimal CPP implementation of IVariableStateInternal (e.g. VariableStateInternal)
 * @ingroup ie_dev_api_variable_state_api
 */
template <class T>
class VariableStateBase : public IVariableState {
    std::shared_ptr<T> impl;

public:
    /**
     * @brief Constructor with actual underlying implementation.
     * @param impl Underlying implementation of type IVariableStateInternal
     */
    explicit VariableStateBase(std::shared_ptr<T> impl): impl(impl) {
        if (impl == nullptr) {
            THROW_IE_EXCEPTION << "VariableStateBase implementation is not defined";
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

}  // namespace InferenceEngine
