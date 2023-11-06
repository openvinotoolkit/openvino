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

class IVariableState : public ov::IVariableState {
public:
    using ov::IVariableState::IVariableState;

    virtual void Commit() = 0;

    virtual MemoryPtr InputMem() = 0;
    virtual MemoryPtr OutputMem() = 0;
    virtual MemoryDescPtr InternalDesc() const = 0;
};

class VariableStateDoubleBuffer : public IVariableState {
public:
    using MemBuilder = std::function<MemoryPtr(void)>;

public:
    VariableStateDoubleBuffer(std::string name,
                              const MemBuilder& mem_build,
                              MemoryDescPtr external_desc,
                              MemoryCPtr init_val);
    //InferenceEngine::IVariableStateInternal
    void Reset() override;
    void SetState(const InferenceEngine::Blob::Ptr& newState) override;
    InferenceEngine::Blob::CPtr GetState() const override;

    //ov::intel_cpu::IVariableState
    void Commit() override;

    MemoryPtr InputMem() override;
    MemoryPtr OutputMem() override;
    MemoryDescPtr InternalDesc() const override;

private:
    static MemoryDescPtr ToStatic(const MemoryDescPtr& desc);

    void ResetPrimeMem(const MemoryPtr& mem) {
        m_internal_mem[buffer_num] = mem;
    }

    void ResetSecondMem(const MemoryPtr& mem) {
        m_internal_mem[buffer_num ^ 0x1] = mem;
    }

    const MemoryPtr& PrimeMem() const {
        return m_internal_mem[buffer_num];
    }

    const MemoryPtr& SecondMem() const {
        return m_internal_mem[buffer_num ^ 0x1];
    }


    const dnnl::engine& getEngine() const;

private:
    MemoryDescPtr m_external_desc;
    MemoryDescPtr m_internal_desc; //mem desc required by the graph internal tensor
    std::array<MemoryPtr, 2> m_internal_mem{};
    size_t buffer_num = 0;
};

using MemStatePtr = std::shared_ptr<IVariableState>;
using MemStateCPtr = std::shared_ptr<const IVariableState>;

}   // namespace intel_cpu
}   // namespace ov
