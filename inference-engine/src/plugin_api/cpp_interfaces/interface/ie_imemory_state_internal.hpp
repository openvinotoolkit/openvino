// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

#include <memory>
#include <string>

namespace InferenceEngine {
/**
 * @interface IVariableStateInternal
 * @brief minimal interface for memory state implementation
 * @ingroup ie_dev_api_mem_state_api
 */
class IVariableStateInternal {
public:
    using Ptr = std::shared_ptr<IVariableStateInternal>;

    virtual ~IVariableStateInternal() = default;
    virtual std::string GetName() const = 0;
    virtual void Reset() = 0;
    virtual void SetState(Blob::Ptr newState) = 0;
    virtual Blob::CPtr GetState() const = 0;
    INFERENCE_ENGINE_DEPRECATED("Use GetState function instead")
    virtual Blob::CPtr GetLastState() const {return GetState();}
};

/*
 * @brief For compatibility reasons.
 */
using IMemoryStateInternal = IVariableStateInternal;
}  // namespace InferenceEngine
