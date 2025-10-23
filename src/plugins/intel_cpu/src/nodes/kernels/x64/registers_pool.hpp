// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu {

/**
 * The RegistersPool is the base class for the IsaRegistersPool template:
 *      template <x64::cpu_isa_t isa>
 *      class IsaRegistersPool : public RegistersPool;
 *
 * The registers pool must be created by instantiating the IsaRegistersPool template, like the next:
 *      RegistersPool::Ptr regPool = RegistersPool::create<isa>({
 *          // the list of the registers to be excluded from pool
 *          Reg64(Operand::RAX), Reg64(Operand::RCX), Reg64(Operand::RDX), Reg64(Operand::RBX),
 *          Reg64(Operand::RSP), Reg64(Operand::RBP), Reg64(Operand::RSI), Reg64(Operand::RDI)
 *      });
 */
class RegistersPool {
public:
    using Ptr = std::shared_ptr<RegistersPool>;
    using WeakPtr = std::weak_ptr<RegistersPool>;
    static constexpr int anyIdx = -1;

    /**
     * The scoped wrapper for the Xbyak registers.
     * By creating it you are getting the register from the pool RegistersPool.
     * It could be created by using constructor with RegistersPool as an argument, like the next:
     *      const RegistersPool::Reg<Xbyak::Reg64> reg {regPool};
     * The destructor will return the register to the pool. Or it could be returned manually:
     *      reg.release();
     * @tparam TReg Xbyak register class
     */
    template <typename TReg>
    class Reg {
        friend class RegistersPool;

    public:
        Reg() = default;
        explicit Reg(const RegistersPool::Ptr& regPool) {
            initialize(regPool);
        }
        Reg(const RegistersPool::Ptr& regPool, int requestedIdx) {
            initialize(regPool, requestedIdx);
        }
        ~Reg() {
            release();
        }
        Reg& operator=(Reg&& other) noexcept {
            release();
            reg = other.reg;
            regPool = std::move(other.regPool);
            return *this;
        }
        Reg(Reg&& other) noexcept : reg(other.reg), regPool(std::move(other.regPool)) {}
        // XByak relies on implicit conversions
        operator TReg&() {  // NOLINT(google-explicit-constructor)
            ensureValid();
            return reg;
        }
        operator const TReg&() const {  // NOLINT(google-explicit-constructor)
            ensureValid();
            return reg;
        }
        operator Xbyak::RegExp() const {  // NOLINT(google-explicit-constructor)
            ensureValid();
            return reg;
        }
        [[nodiscard]] int getIdx() const {
            ensureValid();
            return reg.getIdx();
        }
        friend Xbyak::RegExp operator+(const Reg& lhs, const Xbyak::RegExp& rhs) {
            lhs.ensureValid();
            return lhs.operator Xbyak::RegExp() + rhs;
        }
        void release() {
            if (auto pool = regPool.lock()) {
                pool->returnToPool(reg);
                regPool.reset();
            }
        }
        [[nodiscard]] bool isInitialized() const {
            return !regPool.expired();
        }

    private:
        void ensureValid() const {
            OPENVINO_ASSERT(isInitialized(), "RegistersPool::Reg is either not initialized or released");
        }

        void initialize(const RegistersPool::Ptr& pool, int requestedIdx = anyIdx) {
            static_assert(is_any_of<TReg,
                                    Xbyak::Xmm,
                                    Xbyak::Ymm,
                                    Xbyak::Zmm,
                                    Xbyak::Reg8,
                                    Xbyak::Reg16,
                                    Xbyak::Reg32,
                                    Xbyak::Reg64,
                                    Xbyak::Opmask>::value,
                          "Unsupported TReg by RegistersPool::Reg. Please, use the following Xbyak registers either "
                          "Reg8, Reg16, Reg32, Reg64, Xmm, Ymm, Zmm or Opmask");
            release();
            reg = TReg(pool->template getFree<TReg>(requestedIdx));
            regPool = pool;
        }

        TReg reg;
        RegistersPool::WeakPtr regPool;
    };

    virtual ~RegistersPool() {
        checkUniqueAndUpdate(false);
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    static Ptr create(std::initializer_list<Xbyak::Reg> regsToExclude);

    static Ptr create(dnnl::impl::cpu::x64::cpu_isa_t isa, std::initializer_list<Xbyak::Reg> regsToExclude);

    template <typename TReg>
    [[nodiscard]] size_t countFree() const {
        static_assert(is_any_of<TReg,
                                Xbyak::Xmm,
                                Xbyak::Ymm,
                                Xbyak::Zmm,
                                Xbyak::Reg8,
                                Xbyak::Reg16,
                                Xbyak::Reg32,
                                Xbyak::Reg64,
                                Xbyak::Opmask>::value,
                      "Unsupported TReg by RegistersPool::Reg. Please, use the following Xbyak registers either "
                      "Reg8, Reg16, Reg32, Reg64, Xmm, Ymm, Zmm or Opmask");
        if (std::is_base_of_v<Xbyak::Mmx, TReg>) {
            return simdSet.countUnused();
        }
        if (std::is_same_v<TReg, Xbyak::Reg8> || std::is_same_v<TReg, Xbyak::Reg16> ||
            std::is_same_v<TReg, Xbyak::Reg32> || std::is_same_v<TReg, Xbyak::Reg64>) {
            return generalSet.countUnused();
        }
        if (std::is_same_v<TReg, Xbyak::Opmask>) {
            return countUnusedOpmask();
        }
    }

protected:
    class PhysicalSet {
    public:
        explicit PhysicalSet(int size) : isFreeIndexVector(size, true) {}

        void setAsUsed(size_t regIdx) {
            OPENVINO_ASSERT(regIdx < isFreeIndexVector.size(),
                            "regIdx is out of bounds in RegistersPool::PhysicalSet::setAsUsed()");
            OPENVINO_ASSERT(isFreeIndexVector[regIdx], "Inconsistency in RegistersPool::PhysicalSet::setAsUsed()");
            isFreeIndexVector[regIdx] = false;
        }

        void setAsUnused(size_t regIdx) {
            OPENVINO_ASSERT(regIdx < isFreeIndexVector.size(),
                            "regIdx is out of bounds in RegistersPool::PhysicalSet::setAsUsed()");
            OPENVINO_ASSERT(!isFreeIndexVector[regIdx], "Inconsistency in RegistersPool::PhysicalSet::setAsUnused()");
            isFreeIndexVector[regIdx] = true;
        }

        size_t getUnused(size_t requestedIdx) {
            if (requestedIdx == static_cast<size_t>(anyIdx)) {
                return getFirstFreeIndex();
            }
            OPENVINO_ASSERT(requestedIdx < isFreeIndexVector.size(),
                            "requestedIdx is out of bounds in RegistersPool::PhysicalSet::getUnused()");
            OPENVINO_ASSERT(isFreeIndexVector[requestedIdx],
                            "The register with index #",
                            requestedIdx,
                            " already used in the RegistersPool");
            return requestedIdx;
        }

        void exclude(Xbyak::Reg reg) {
            isFreeIndexVector.at(reg.getIdx()) = false;
        }

        [[nodiscard]] size_t countUnused() const {
            size_t count = 0;
            for (const auto& isFree : isFreeIndexVector) {
                if (isFree) {
                    ++count;
                }
            }
            return count;
        }

    private:
        size_t getFirstFreeIndex() {
            for (size_t c = 0; c < isFreeIndexVector.size(); ++c) {
                if (isFreeIndexVector[c]) {
                    return c;
                }
            }
            OPENVINO_THROW("Not enough registers in the RegistersPool");
        }

        std::vector<bool> isFreeIndexVector;
    };

    virtual int getFreeOpmask([[maybe_unused]] int requestedIdx) {
        OPENVINO_THROW("getFreeOpmask: The Opmask is not supported in current instruction set");
    }
    virtual void returnOpmaskToPool([[maybe_unused]] int idx) {
        OPENVINO_THROW("returnOpmaskToPool: The Opmask is not supported in current instruction set");
    }
    [[nodiscard]] virtual size_t countUnusedOpmask() const {
        OPENVINO_THROW("countUnusedOpmask: The Opmask is not supported in current instruction set");
    }

    explicit RegistersPool(int simdRegistersNumber) : simdSet(simdRegistersNumber) {
        checkUniqueAndUpdate();
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RAX));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RCX));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RDI));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RBP));
    }

    RegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude, int simdRegistersNumber)
        : simdSet(simdRegistersNumber) {
        checkUniqueAndUpdate();
        for (const auto& reg : regsToExclude) {
            if (reg.isXMM() || reg.isYMM() || reg.isZMM()) {
                simdSet.exclude(reg);
            } else if (reg.isREG()) {
                generalSet.exclude(reg);
            }
        }
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
    }

private:
    template <typename TReg>
    int getFree(int requestedIdx) {
        if (std::is_base_of_v<Xbyak::Mmx, TReg>) {
            auto idx = simdSet.getUnused(requestedIdx);
            simdSet.setAsUsed(idx);
            return idx;
        }
        if (std::is_same_v<TReg, Xbyak::Reg8> || std::is_same_v<TReg, Xbyak::Reg16> ||
            std::is_same_v<TReg, Xbyak::Reg32> || std::is_same_v<TReg, Xbyak::Reg64>) {
            auto idx = generalSet.getUnused(requestedIdx);
            generalSet.setAsUsed(idx);
            return idx;
        }
        if (std::is_same_v<TReg, Xbyak::Opmask>) {
            return getFreeOpmask(requestedIdx);
        }
    }

    template <typename TReg>
    void returnToPool(const TReg& reg) {
        if (std::is_base_of_v<Xbyak::Mmx, TReg>) {
            simdSet.setAsUnused(reg.getIdx());
        } else if (std::is_same_v<TReg, Xbyak::Reg8> || std::is_same_v<TReg, Xbyak::Reg16> ||
                   std::is_same_v<TReg, Xbyak::Reg32> || std::is_same_v<TReg, Xbyak::Reg64>) {
            generalSet.setAsUnused(reg.getIdx());
        } else if (std::is_same_v<TReg, Xbyak::Opmask>) {
            returnOpmaskToPool(reg.getIdx());
        }
    }

    static void checkUniqueAndUpdate(bool isCtor = true) {
        static thread_local bool isCreated = false;
        if (isCtor) {
            OPENVINO_ASSERT(!isCreated, "There should be only one instance of RegistersPool per thread");
            isCreated = true;
        } else {
            isCreated = false;
        }
    }

    PhysicalSet generalSet{16};
    PhysicalSet simdSet;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class IsaRegistersPool : public RegistersPool {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude)
        : RegistersPool(regsToExclude, dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::n_vregs) {}
};

template <>
class IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core> : public RegistersPool {
public:
    IsaRegistersPool()
        : RegistersPool(dnnl::impl::cpu::x64::cpu_isa_traits_t<dnnl::impl::cpu::x64::avx512_core>::n_vregs) {
        opmaskSet.exclude(
            Xbyak::Opmask(0));  // the Opmask(0) has special meaning for some instructions, like gather instruction
    }

    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude)
        : RegistersPool(regsToExclude,
                        dnnl::impl::cpu::x64::cpu_isa_traits_t<dnnl::impl::cpu::x64::avx512_core>::n_vregs) {
        for (const auto& reg : regsToExclude) {
            if (reg.isOPMASK()) {
                opmaskSet.exclude(reg);
            }
        }
    }

    int getFreeOpmask(int requestedIdx) override {
        auto idx = opmaskSet.getUnused(requestedIdx);
        opmaskSet.setAsUsed(idx);
        return idx;
    }

    void returnOpmaskToPool(int idx) override {
        opmaskSet.setAsUnused(idx);
    }

    [[nodiscard]] size_t countUnusedOpmask() const override {
        return opmaskSet.countUnused();
    }

protected:
    PhysicalSet opmaskSet{8};
};

template <>
class IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core_vnni>
    : public IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core> {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude)
        : IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core>(regsToExclude) {}
    IsaRegistersPool() = default;
};

template <>
class IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core_bf16>
    : public IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core> {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude)
        : IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core>(regsToExclude) {}
    IsaRegistersPool() = default;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
RegistersPool::Ptr RegistersPool::create(std::initializer_list<Xbyak::Reg> regsToExclude) {
    return std::make_shared<IsaRegistersPool<isa>>(regsToExclude);
}

inline RegistersPool::Ptr RegistersPool::create(dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                std::initializer_list<Xbyak::Reg> regsToExclude) {
#define ISA_SWITCH_CASE(isa) \
    case isa:                \
        return std::make_shared<IsaRegistersPool<(isa)>>(regsToExclude);
    switch (isa) {
        ISA_SWITCH_CASE(dnnl::impl::cpu::x64::sse41)
        ISA_SWITCH_CASE(dnnl::impl::cpu::x64::avx)
        ISA_SWITCH_CASE(dnnl::impl::cpu::x64::avx2)
        ISA_SWITCH_CASE(dnnl::impl::cpu::x64::avx2_vnni)
        ISA_SWITCH_CASE(dnnl::impl::cpu::x64::avx512_core)
        ISA_SWITCH_CASE(dnnl::impl::cpu::x64::avx512_core_vnni)
        ISA_SWITCH_CASE(dnnl::impl::cpu::x64::avx512_core_bf16)
        ISA_SWITCH_CASE(dnnl::impl::cpu::x64::avx512_core_fp16)
    case dnnl::impl::cpu::x64::avx512_core_bf16_ymm:
        return std::make_shared<IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core>>(regsToExclude);
    case dnnl::impl::cpu::x64::avx512_core_amx:
        return std::make_shared<IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core>>(regsToExclude);
    case dnnl::impl::cpu::x64::avx512_vpopcnt:
        return std::make_shared<IsaRegistersPool<dnnl::impl::cpu::x64::avx512_core>>(regsToExclude);
    case dnnl::impl::cpu::x64::isa_undef:
    case dnnl::impl::cpu::x64::amx_tile:
    case dnnl::impl::cpu::x64::amx_int8:
    case dnnl::impl::cpu::x64::amx_bf16:
    case dnnl::impl::cpu::x64::avx2_vnni_2:
    case dnnl::impl::cpu::x64::amx_fp16:
    case dnnl::impl::cpu::x64::avx512_core_amx_fp16:
    case dnnl::impl::cpu::x64::isa_all:
        OPENVINO_THROW("Invalid isa argument in RegistersPool::create()");
    }
    OPENVINO_THROW("Invalid isa argument in RegistersPool::create()");
#undef ISA_SWITCH_CASE
}

}  // namespace ov::intel_cpu
