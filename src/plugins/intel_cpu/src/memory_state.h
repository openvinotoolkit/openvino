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

    virtual void commit() = 0;

    virtual MemoryPtr input_mem() const = 0;
    virtual MemoryPtr output_mem() const = 0;
    virtual MemoryDescPtr internal_desc() const = 0;
};

class VariableStateBase : public IVariableState {
public:
    VariableStateBase(const std::string& name, const MemoryDescPtr& external_desc);

    //ov::IVariableState
    void set_state(const ov::SoPtr<ov::ITensor>& state) override;
    ov::SoPtr<ov::ITensor> get_state() const override;

protected:
    static MemoryDescPtr to_static(const MemoryDescPtr& desc);
    static const dnnl::engine& get_engine();

protected:
    MemoryDescPtr m_external_desc;
};

class VariableStateDoubleBuffer : public VariableStateBase {
public:
    VariableStateDoubleBuffer(const std::string& name,
                              const MemoryPtr& first_buffer,
                              const MemoryPtr& second_buffer,
                              const MemoryDescPtr& external_desc,
                              const MemoryCPtr& init_val);

    //ov::IVariableState
    void reset() override;

    //ov::intel_cpu::IVariableState
    void commit() override;

    MemoryPtr input_mem() const override;
    MemoryPtr output_mem() const override;
    MemoryDescPtr internal_desc() const override;

private:
    void reset_prime_mem(const MemoryPtr& mem) {
        m_internal_mem[buffer_num] = mem;
    }

    void reset_second_mem(const MemoryPtr& mem) {
        m_internal_mem[buffer_num ^ 0x1] = mem;
    }

    const MemoryPtr& prime_mem() const {
        return m_internal_mem[buffer_num];
    }

    const MemoryPtr& second_mem() const {
        return m_internal_mem[buffer_num ^ 0x1];
    }

private:
    MemoryDescPtr m_internal_desc; //mem desc required by the graph internal tensor
    std::array<MemoryPtr, 2> m_internal_mem{};
    size_t buffer_num = 0;
};

class VariableStateSingleBuffer : public VariableStateBase {
public:
    VariableStateSingleBuffer(const std::string& name,
                              const MemoryPtr& buffer,
                              const MemoryDescPtr& external_desc,
                              const MemoryCPtr& init_val);
    //ov::IVariableState
    void reset() override;

    //ov::intel_cpu::IVariableState
    void commit() override;

    MemoryPtr input_mem() const override;
    MemoryPtr output_mem() const override;
    MemoryDescPtr internal_desc() const override;

private:
    MemoryDescPtr m_internal_desc; //mem desc required by the graph internal tensor
    MemoryPtr m_internal_mem;
};

using MemStatePtr = std::shared_ptr<IVariableState>;
using MemStateCPtr = std::shared_ptr<const IVariableState>;
}   // namespace intel_cpu
}   // namespace ov
