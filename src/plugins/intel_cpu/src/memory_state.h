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
    VariableState(std::string name, MemoryPtr storage)
        : InferenceEngine::IVariableStateInternal{name} {
        tensor_desc = MemoryDescUtils::convertToTensorDesc(storage->getDesc());
    }

    void reset() override;

private:
    InferenceEngine::TensorDesc tensor_desc;  // shape of initial state
};

}  // namespace intel_cpu
}  // namespace ov
