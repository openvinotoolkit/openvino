// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides VariableState
 *
 * @file ie_memory_state.hpp
 */

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <memory>
#include <string>

#include "ie_api.h"
#include "ie_blob.h"

namespace InferenceEngine {

class IVariableStateInternal;
IE_SUPPRESS_DEPRECATED_START

/**
 * @brief VariableState class
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(VariableState) {
    std::shared_ptr<IVariableStateInternal> _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs VariableState from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Optional: Plugin to use. This is required to ensure that VariableState can work properly even if plugin
     * object is destroyed.
     */
    VariableState(const std::shared_ptr<IVariableStateInternal>& impl, const std::shared_ptr<void>& so);
    friend class InferRequest;
    friend class ExecutableNetwork;

public:
    /// @brief Default constructor
    VariableState() = default;

    /// @brief Default copy constructor
    /// @param other other VariableState object
    VariableState(const VariableState& other) = default;

    /// @brief Default copy assignment operator
    /// @param other other VariableState object
    /// @return reference to the current object
    VariableState& operator=(const VariableState& other) = default;

    /// @brief Default move constructor
    /// @param other other VariableState object
    VariableState(VariableState&& other) = default;

    /// @brief Default move assignment operator
    /// @param other other VariableState object
    /// @return reference to the current object
    VariableState& operator=(VariableState&& other) = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~VariableState();

    /**
     * @brief Reset internal variable state for relevant infer request,
     * to a value specified as default for according ReadValue node
     */
    void Reset();

    /**
     * @brief Gets name of current variable state, if length of array is not enough name is truncated by len, null
     * terminator is inserted as well. As variable state name `variable_id` from according `ReadValue` used.
     * @return A string representing a state name
     */
    std::string GetName() const;

    /**
     * @brief Returns the value of the variable state.
     * @return A blob representing a state
     */
    Blob::CPtr GetState() const;

    /**
     * @brief Sets the new state for the next inference.
     * @param state The current state to set
     */
    void SetState(Blob::Ptr state);
};

/**
 * @brief For compatibility reasons.
 */
using MemoryState = VariableState;

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
