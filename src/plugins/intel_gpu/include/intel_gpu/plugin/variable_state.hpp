// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include "intel_gpu/plugin/graph.hpp"
#include <functional>

namespace ov {
namespace runtime {
namespace intel_gpu {

class VariableState : public InferenceEngine::IVariableStateInternal {
public:
    VariableState(const std::string& name, const std::vector<cldnn::network::VariableState::Ptr>& states,
                  std::shared_ptr<cldnn::engine> engine, int currentBatch);

    /**
     * @brief Reset internal variable state for relevant infer request, to a value specified as
     * default for according `ReadValue` node
     */
    void Reset() override;

    /**
     * @brief Sets the new state for the next inference
     * @param newState A new state
     */
    void SetState(const InferenceEngine::Blob::Ptr &newState) override;

    /**
     * @brief Returns the value of the variable state.
     * @return The value of the variable state
     */
    InferenceEngine::Blob::CPtr GetState() const override;

protected:
    InferenceEngine::SizeVector AggregateShape(const cldnn::layout &layout);
    void IterateOverStates(std::function<void(cldnn::network::VariableState&)> f) const;

private:
    int currentBatch_;
    std::vector<cldnn::network::VariableState::Ptr> states_;
    InferenceEngine::TensorDesc desc_;
    std::shared_ptr<cldnn::engine> engine_;
};

} // namespace intel_gpu
} // namespace runtime
} // namespace ov
