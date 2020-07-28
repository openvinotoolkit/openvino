// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "cpp_interfaces/exception2status.hpp"
#include "ie_imemory_state.hpp"

namespace InferenceEngine {

/**
 * @brief default implementation for IMemoryState
 * @ingroup ie_dev_api_mem_state_api
 */
template <class T>
class MemoryStateBase : public IMemoryState {
protected:
    std::shared_ptr<T> impl;

public:
    explicit MemoryStateBase(std::shared_ptr<T> impl): impl(impl) {
        if (impl == nullptr) {
            THROW_IE_EXCEPTION << "MemoryStateBase implementation not defined";
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

    StatusCode GetLastState(Blob::CPtr& lastState, ResponseDesc* resp) const noexcept override {
        TO_STATUS(lastState = impl->GetLastState());
    }
};

}  // namespace InferenceEngine