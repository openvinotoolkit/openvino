// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ie_blob.h"
#include "so_ptr.hpp"

namespace InferenceEngine {

/**
 * @interface IVariableStateInternal
 * @brief Minimal interface for variable state implementation
 * @ingroup ie_dev_api_variable_state_api
 */
class INFERENCE_ENGINE_API_CLASS(IVariableStateInternal) : public std::enable_shared_from_this<IVariableStateInternal> {
public:
    /**
     * @brief A shared pointer to a IVariableStateInternal interface
     */
    using Ptr = std::shared_ptr<IVariableStateInternal>;

    explicit IVariableStateInternal(const std::string& name);

    /**
     * @brief Gets a variable state name
     * @return A string representing variable state name
     */
    virtual std::string GetName() const;

    /**
     * @brief Reset internal variable state for relevant infer request, to a value specified as
     * default for according `ReadValue` node
     */
    virtual void Reset();

    /**
     * @brief Sets the new state for the next inference
     * @param newState A new state
     */
    virtual void SetState(const Blob::Ptr& newState);

    /**
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    virtual Blob::CPtr GetState() const;

protected:
    /**
     * @brief A default dtor
     */
    ~IVariableStateInternal() = default;

    std::string name;
    Blob::Ptr state;
};

/**
 * @brief For compatibility reasons.
 */
using IMemoryStateInternal = IVariableStateInternal;

/**
 * @brief SoPtr to IVariableStateInternal.
 */
using SoIVariableStateInternal = ov::SoPtr<IVariableStateInternal>;

/**
 * @brief For compatibility reasons.
 */
using MemoryStateInternal = IVariableStateInternal;

}  // namespace InferenceEngine

namespace ov {

class ITensor;

/**
 * @interface IVariableState
 * @brief Minimal interface for variable state implementation
 * @ingroup ie_dev_api_variable_state_api
 */
class OPENVINO_API IVariableState : public std::enable_shared_from_this<IVariableState> {
public:
    /**
     * @brief A shared pointer to a IVariableStateInternal interface
     */
    using Ptr = std::shared_ptr<IVariableState>;

    explicit IVariableState(const std::string& name);

    /**
     * @brief Gets a variable state name
     * @return A string representing variable state name
     */
    virtual std::string get_name() const;

    /**
     * @brief Reset internal variable state for relevant infer request, to a value specified as
     * default for according `ReadValue` node
     */
    virtual void reset();

    /**
     * @brief Sets the new state for the next inference
     * @param newState A new state
     */
    virtual void set_state(const std::shared_ptr<ITensor>& new_state);

    /**
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    virtual std::shared_ptr<ITensor> get_state() const;

    std::string name;
    std::shared_ptr<ITensor> state;
};

struct OPENVINO_API IEVariableState : public IVariableState {
    explicit IEVariableState(const InferenceEngine::IVariableStateInternal::Ptr& impl_);
    std::string get_name() const override;
    void reset() override;
    void set_state(const std::shared_ptr<ITensor>& new_state) override;
    std::shared_ptr<ITensor> get_state() const override;
    ie::IVariableStateInternal::Ptr impl;
};

}  // namespace ov