// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides VariableState
 *
 * @file variable_state.hpp
 */

#pragma once

#include <ie_api.h>
#include <ie_blob.h>

#include <memory>
#include <string>

#include "common.hpp"

namespace InferenceEngine {
class IVariableStateInternal;
class Blob;
}  // namespace InferenceEngine

namespace ov {
namespace runtime {

class SharedObject;
class InferRequest;

/**
 * @brief VariableState class
 */
class INFERENCE_ENGINE_API_CLASS(VariableState) {
    std::shared_ptr<SharedObject> _so;
    std::shared_ptr<ie::IVariableStateInternal> _impl;

    /**
     * @brief Constructs VariableState from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Optional: Plugin to use. This is required to ensure that VariableState can work properly even if plugin
     * object is destroyed.
     */
    VariableState(const std::shared_ptr<SharedObject>& so, const std::shared_ptr<ie::IVariableStateInternal>& impl);

    friend class ov::runtime::InferRequest;

public:
    /**
     * @brief Default constructor
     */
    VariableState() = default;

    /**
     * @brief Reset internal variable state for relevant infer request,
     * to a value specified as default for according ReadValue node
     */
    void reset();

    /**
     * @brief Gets name of current variable state, if length of array is not enough name is truncated by len, null
     * terminator is inserted as well. As variable state name `variable_id` from according `ReadValue` used.
     * @return A string representing a state name
     */
    std::string get_name() const;

    /**
     * @brief Returns the value of the variable state.
     * @return A blob representing a state
     */
    std::shared_ptr<const ie::Blob> get_state() const;

    /**
     * @brief Sets the new state for the next inference.
     * @param state The current state to set
     */
    void set_state(const std::shared_ptr<ie::Blob>& state);
};
}  // namespace runtime
}  // namespace ov
