// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_base.hpp"
#include "emitters/jit_emitter.hpp"
#include "registers_pool.hpp"
#include "stack_allocator.hpp"

namespace ov {
namespace intel_cpu {

#define getReg64() RegistersPool::Reg<Xbyak::Reg64>(registersPool)
#define getVmm()   RegistersPool::Reg<Vmm>(registersPool)
#define getMask()  RegistersPool::Reg<Vmask>(registersPool)

class jit_kernel_base: public jit_base {
public:
    jit_kernel_base(const char* name, x64::cpu_isa_t max_cpu_isa)
        : jit_base(name, max_cpu_isa) {}

    void generate() override;
    virtual void generate_impl() = 0;

    void emu_vscatterdps(const Xbyak::Reg64& reg_addr,
                         const Xbyak::Xmm& xmm_index,
                         const Xbyak::Xmm& xmm_val,
                         const Xbyak::Reg& reg_mask,
                         const bool is_mask_seq = true);

    void gatherdd(const Xbyak::Xmm&    vDst,
                  const Xbyak::Reg64&  rSrcPtr,
                  const Xbyak::Xmm&    vSrcShift,
                  const Xbyak::Opmask& kReadMask,
                  const bool useMask   = true,
                  const bool zeroFill  = false);

    void gatherdd(const Xbyak::Xmm&   vDst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Xmm&   vSrcShift,
                  const Xbyak::Xmm&   vReadMask,
                  const bool useMask  = true,
                  const bool zeroFill = false);

    void gatherdd(const Xbyak::Ymm&   vDst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Ymm&   vSrcShift,
                  const Xbyak::Ymm&   vReadMask,
                  const bool useMask  = true,
                  const bool zeroFill = false);

    void fillRestWorkMask(const Xbyak::Opmask& kDstMask,
                          const Xbyak::Zmm& zAux,
                          const Xbyak::Reg64& rWorkRest);

    // Makes gather from memory under the vReadMask and writes to the memory m128.
    void memMovDD(const Xbyak::Reg64& rDst,
                  const Xbyak::Reg64& rSrc,
                  const Xbyak::Xmm&   vReadMask,
                  const Xbyak::Xmm&   vSrcShift,
                  const Xbyak::Reg64& rToStoreCounter,
                  const bool useMask  = true,
                  const bool zeroFill = false);

    // Makes gather from the memory under the vReadMask and writes to the memory m256.
    void memMovDD(const Xbyak::Reg64& rDst,
                  const Xbyak::Reg64& rSrc,
                  const Xbyak::Ymm&   vReadMask,
                  const Xbyak::Ymm&   vSrcShift,
                  const Xbyak::Reg64& rToStoreCounter,
                  const bool useMask  = true,
                  const bool zeroFill = false);

protected:
    virtual void createRegistersPool();
    virtual void createStackAllocator();

    RegistersPool::Ptr registersPool;
    std::unique_ptr<StackAllocator> stackAllocator;
    std::unordered_map<std::string, std::shared_ptr<jit_emitter>> emittersMap;
};

} // namespace intel_cpu
} // namespace ov
