// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include "registers_pool.hpp"

namespace ov {
namespace intel_cpu {

using namespace dnnl::impl::cpu;

template<typename TReg>
class ForceReg {
    friend class RegistersPool;
public:
    ForceReg(x64::jit_generator& code_gen,
             std::vector<Xbyak::Reg>& not_available_reg,
             const RegistersPool::Ptr& regPool)
             : code_gen{code_gen}
             , not_available_reg{not_available_reg} {
        try {
            rp_reg = RegistersPool::Reg<TReg>{regPool};
            reg = rp_reg;
        } catch (...) {
            reg = regPool->getInplaceFree<TReg>(not_available_reg);
            code_gen.push(reg);
            not_available_reg.push_back(reg);
        }
    }
    ForceReg(x64::jit_generator& code_gen,
             std::vector<Xbyak::Reg>& not_available_reg,
             const RegistersPool::Ptr& regPool,
             int requestedIdx) {
        try {
            rp_reg = RegistersPool::Reg<TReg>{regPool, requestedIdx};
            reg = rp_reg;
        } catch (...) {
            reg = regPool->getInplaceFree<TReg>(not_available_reg);
            code_gen.push(reg);
            not_available_reg.push_back(reg);
        }
    }
    ~ForceReg() {
        if (!rp_reg.isInitialized()) {
            code_gen.pop(reg);
            auto found = std::find(not_available_reg.begin(), not_available_reg.end(), reg);
            not_available_reg.erase(found);
        }
    }
    operator TReg&() { return reg; }
    operator const TReg&() const { return reg; }
    operator Xbyak::RegExp() const { return reg; }
    ForceReg& operator=(const StackAllocator::Address& addr) {
        stack_mov(*this, addr);
        return *this;
    }
    int getIdx() const { return rp_reg.getIdx(); }
    friend Xbyak::RegExp operator+(const ForceReg& lhs, const Xbyak::RegExp& rhs) {
        return lhs.operator Xbyak::RegExp() + rhs;
    }

private:
    x64::jit_generator& code_gen;
    std::vector<Xbyak::Reg>& not_available_reg;
    RegistersPool::Reg<TReg> rp_reg{};
    TReg reg{};
};

} // namespace intel_cpu
} // namespace ov
