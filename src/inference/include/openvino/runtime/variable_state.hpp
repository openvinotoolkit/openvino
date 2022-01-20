// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides ov::runtime::VariableState
 * @file openvino/runtime/variable_state.hpp
 */

#pragma once

#include <memory>
#include <string>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/tensor.hpp"

namespace InferenceEngine {
class IVariableStateInternal;
}  // namespace InferenceEngine

namespace ov {
namespace runtime {

class InferRequest;

/**
 * @brief VariableState class
 */
class OPENVINO_RUNTIME_API VariableState {
    std::shared_ptr<InferenceEngine::IVariableStateInternal> _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs VariableState from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Optional: Plugin to use. This is required to ensure that VariableState can work properly even if plugin
     * object is destroyed.
     */
    VariableState(const std::shared_ptr<InferenceEngine::IVariableStateInternal>& impl,
                  const std::shared_ptr<void>& so);

    friend class ov::runtime::InferRequest;

public:
    /**
     * @brief Default constructor
     */
    VariableState() = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~VariableState();

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
     * @return A tensor representing a state
     */
    Tensor get_state() const;

    /**
     * @brief Sets the new state for the next inference.
     * @param state The current state to set
     */
    void set_state(const Tensor& state);
};
}  // namespace runtime
}  // namespace ov
