// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "memory_desc/blocked_memory_desc.h"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/tensor.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {

class IVariableState : public ov::IVariableState {
public:
    using ov::IVariableState::IVariableState;

    virtual void commit() = 0;

    virtual MemoryPtr input_mem() = 0;
    virtual MemoryPtr output_mem() = 0;
    virtual MemoryDescPtr internal_desc() const = 0;
    virtual bool is_reset_state() const = 0;
};

class VariableStateBase : public IVariableState {
public:
    VariableStateBase(const std::string& name, const MemoryDescPtr& external_desc);

    //ov::IVariableState
    void set_state(const ov::SoPtr<ov::ITensor>& state) override final; // NOLINT
    ov::SoPtr<ov::ITensor> get_state() const override;
    void reset() override final; // NOLINT
    bool is_reset_state() const override final; // NOLINT
    void commit() override final; // NOLINT

protected:
    virtual MemoryPtr internal_state_mem() const = 0;
    virtual void reset_impl() = 0;
    virtual void commit_impl() = 0;
    virtual void set_state_impl(const ov::SoPtr<ov::ITensor>& state);

    static MemoryDescPtr to_static(const MemoryDescPtr& desc);
    static const dnnl::engine& get_engine();

    MemoryDescPtr get_external_desc() const {
        return m_external_desc;
    }

private:
    MemoryDescPtr m_external_desc;
    bool reset_state_flag = true;
};

class VariableStateDoubleBuffer : public VariableStateBase {
public:
    VariableStateDoubleBuffer(const std::string& name,
                              const MemoryPtr& first_buffer,
                              const MemoryPtr& second_buffer,
                              const MemoryDescPtr& external_desc);

    MemoryPtr input_mem() override;
    MemoryPtr output_mem() override;
    MemoryDescPtr internal_desc() const override;

private:
    //ov::intel_cpu::VariableStateBase
    void reset_impl() override;
    void commit_impl() override;

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

    MemoryPtr internal_state_mem() const override;

private:
    MemoryDescPtr m_internal_desc; //mem desc required by the graph internal tensor
    std::array<MemoryPtr, 2> m_internal_mem{};
    size_t buffer_num = 0;
};

class VariableStateSingleBuffer : public VariableStateBase {
public:
    VariableStateSingleBuffer(const std::string& name,
                              const MemoryPtr& external_buffer,
                              const MemoryDescPtr& external_desc);

    MemoryPtr input_mem() override;
    MemoryPtr output_mem() override;
    MemoryDescPtr internal_desc() const override;

private:
    void reset_impl() override;
    void commit_impl() override;

    MemoryPtr internal_state_mem() const override;

private:
    MemoryDescPtr m_internal_desc; //mem desc required by the graph internal tensor
    MemoryPtr m_internal_mem;
};

class VariableStateKVcache : public VariableStateBase {
public:
    VariableStateKVcache(const std::string& name,
                         const MemoryDescPtr& external_desc,
                         const BlockedMemoryDescPtr& dense_internal_desc);

    //ov::IVariableState
    ov::SoPtr<ov::ITensor> get_state() const override;

    //ov::intel_cpu::VariableStateBase
    MemoryPtr input_mem() override;
    MemoryPtr output_mem() override;
    MemoryDescPtr internal_desc() const override;

    MemoryPtr internal_state_mem() const override;
    void assign_internal_state(const MemoryPtr& mem);

    MemoryPtr hidden_state_mem() const;
    void assign_hidden_state(const MemoryPtr& mem);

    // size in elements count
    size_t internal_state_max_size() const {
        return m_internal_mem_max_size;
    }
    void assign_internal_state_max_size(size_t max_size) {
        m_internal_mem_max_size = max_size;
    }

    size_t hidden_state_max_size() const {
        return m_hidden_state_max_size;
    }
    void assign_hidden_state_max_size(size_t max_size) {
        m_hidden_state_max_size = max_size;
    }

    PlainTensor& get_scale_zp() {
        return m_scale_zp;
    }
    void set_scale_zp(const PlainTensor& t) {
        m_scale_zp = t;
    }

private:
    //ov::intel_cpu::VariableStateBase
    void set_state_impl(const ov::SoPtr<ov::ITensor>& state) override;
    void reset_impl() override;
    void commit_impl() override;

private:
    MemoryPtr m_internal_mem; // kv cache
    MemoryPtr m_hidden_state; // beam access table
    size_t m_internal_mem_max_size = 0;
    size_t m_hidden_state_max_size = 0;

    // this desc stores the internal prc and axis permutation
    BlockedMemoryDescPtr m_dense_internal_desc;

    // for u8 kv cache: [B, H, L, 2], 0 for scale, 1 for zp
    PlainTensor m_scale_zp;
};

using MemStatePtr = std::shared_ptr<IVariableState>;
using MemStateCPtr = std::shared_ptr<const IVariableState>;
}   // namespace intel_cpu
}   // namespace ov
