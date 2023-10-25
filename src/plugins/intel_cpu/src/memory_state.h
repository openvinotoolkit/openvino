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
    VariableState(std::string name,
                  std::function<MemoryPtr(void)> mem_build,
                  MemoryCPtr init_val);

    void reset() override;
    void SetState(const InferenceEngine::Blob::Ptr& newState) override;
    InferenceEngine::Blob::CPtr GetState() const override;

    void SwapBuffer(); //controversial thing, increase probability of error
    MemoryPtr InternalMem() const {
        return m_internal_mem[buffer_num];
    }

    MemoryDescPtr OriginalDesc() const {
        return m_desc;
    }

private:
    static MemoryDescPtr ToStatic(const MemoryDescPtr& desc);
    void ResetMem(const MemoryPtr& mem) {
        m_internal_mem[buffer_num] = mem;
    }

private:
    MemoryDescPtr m_desc; //mem desc required by the graph internal tensor
    std::function<MemoryPtr(void)> m_mem_build;
    std::array<MemoryPtr, 2> m_internal_mem{};
    size_t buffer_num = 0;
};

using MemStatePtr = std::shared_ptr<VariableState>;
using MemStateCPtr = std::shared_ptr<const VariableState>;

}   // namespace intel_cpu
}   // namespace ov
