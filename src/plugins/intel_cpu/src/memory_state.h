// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "cpu_memory.h"
#include "ie_ngraph_utils.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/cpu_memcpy.h"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace intel_cpu {

class VariableState : public ov::IVariableState {
public:
    VariableState(std::string name, MemoryPtr storage) : ov::IVariableState{name} {
        const auto& memDesc = storage->getDesc();
        m_state = ov::make_tensor(InferenceEngine::details::convertPrecision(memDesc.getPrecision()),
                                  memDesc.getShape().getDims());
        cpu_memcpy(m_state->data(), storage->getData(), storage->getSize());
    }

    void reset() override;
};

}  // namespace intel_cpu
}  // namespace ov
