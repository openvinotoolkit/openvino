// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include "gna_plugin.hpp"

namespace  GNAPluginNS {
namespace memory {
class GNAVariableState : public InferenceEngine::IVariableStateInternal {
 public:
    GNAVariableState(std::string name, std::shared_ptr<GNAMemoryLayer> state)
        : InferenceEngine::IVariableStateInternal{name}, state(state) { IE_ASSERT(state != nullptr); }

    void Reset() override;
    void SetState(const InferenceEngine::Blob::Ptr& newState) override;
    InferenceEngine::Blob::CPtr GetState() const override;
    float GetScaleFactor() const;

private:
    std::shared_ptr<GNAMemoryLayer> state;
    std::string name;

/**
 * @brief Returns InferenceEngine::Precision of input of state depending of element size
 * InferenceEngine::Precision::FP32 if element size equals 4
 * InferenceEngine::Precision::I16 if element size equals 2
 * Exception otherwise
 */
    InferenceEngine::Precision getPrecision() const;
};
}  // namespace memory
}  // namespace GNAPluginNS
