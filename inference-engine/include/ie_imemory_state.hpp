// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for IVariableState interface
 *
 * @file ie_imemory_state.hpp
 */

#pragma once

#include <memory>

#include "ie_blob.h"
#include "ie_common.h"

namespace InferenceEngine {

/**
 * @deprecated Use InferenceEngine::VariableState C++ wrapper instead
 * @interface IVariableState
 * @brief Manages data for reset operations
 */
class INFERENCE_ENGINE_DEPRECATED("InferenceEngine::") IVariableState {
public:
    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief A shared pointer to the IVariableState interface
     */
    using Ptr = std::shared_ptr<IVariableState>;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Gets name of current variable state, if length of array is not enough name is truncated by len, null
     * terminator is inserted as well. As variable state name `variable_id` from according `ReadValue` used. 
     *
     * @param name preallocated buffer for receiving name
     * @param len Length of the buffer
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode GetName(char* name, size_t len, ResponseDesc* resp) const noexcept = 0;

    /**
     * @brief Reset internal variable state for relevant infer request, to a value specified as default for according ReadValue node
     *
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success*
     */
    virtual StatusCode Reset(ResponseDesc* resp) noexcept = 0;

    /**
     * @brief  Sets the new state for the next inference.
     *
     * This method can fail if Blob size does not match the internal state size or precision
     *
     * @param  newState The data to use as new state
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode SetState(Blob::Ptr newState, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Returns the value of the variable state.
     *
     * @param state A reference to a blob containing a variable state
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    INFERENCE_ENGINE_DEPRECATED("Use GetState function instead")
    virtual StatusCode GetLastState(Blob::CPtr& state, ResponseDesc* resp) const noexcept {
        return GetState(state, resp);
    }

    /**
     * @brief Returns the value of the variable state.
     *
     * @param state A reference to a blob containing a variable state
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode GetState(Blob::CPtr& state, ResponseDesc* resp) const noexcept = 0;
};

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief For compatibility reasons.
 */
using IMemoryState = IVariableState;

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine