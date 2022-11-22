// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for internal implementation of VariableState
 *
 * @file ivariable_state.hpp
 */
#include <memory>

#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "openvino/runtime/common.hpp"

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
     * @brief A shared pointer to a IVariableState interface
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

protected:
    ~IVariableState() = default;

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
