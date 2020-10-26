// Copyright (C) 2018-2020 Intel Corporation
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
#include "details/ie_no_copy.hpp"

namespace InferenceEngine {

/**
 * @interface IVariableState
 * @brief manages data for reset operations
 */
class IVariableState : public details::no_copy {
public:
    /**
     * @brief A shared pointer to the IVariableState interface
     */
    using Ptr = std::shared_ptr<IVariableState>;

    /**
     * @brief Gets name of current memory state, if length of array is not enough name is truncated by len, null
     * terminator is inserted as well.
     *
     * @param name preallocated buffer for receiving name
     * @param len Length of the buffer
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode GetName(char* name, size_t len, ResponseDesc* resp) const noexcept = 0;

    /**
     * @brief reset internal memory state for relevant iexecutable network, to a value specified in SetState
     *
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success*
     */
    virtual StatusCode Reset(ResponseDesc* resp) noexcept = 0;

    /**
     * @brief  Sets the new state that is used for all future Reset() operations as a base.
     *
     * This method can fail if Blob size does not match the internal state size or precision
     *
     * @param  newState is the data to use as base state
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode SetState(Blob::Ptr newState, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief returns the value of the last memory state.
     *
     * @details Since we roll memory after each infer, we can query the input state always and still get the last state.
     * @param lastState
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     * */
    INFERENCE_ENGINE_DEPRECATED("Use GetState function instead")
    virtual StatusCode GetLastState(Blob::CPtr& state, ResponseDesc* resp) const noexcept {return GetState(state, resp);}
    virtual StatusCode GetState(Blob::CPtr& state, ResponseDesc* resp) const noexcept = 0;
};

/*
 * @brief For compatibility reasons.
 */
using IMemoryState = IVariableState;
}  // namespace InferenceEngine