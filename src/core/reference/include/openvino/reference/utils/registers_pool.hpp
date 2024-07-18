// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_generator.hpp"
#include "openvino/core/except.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace ov {
namespace runtime {
namespace jit {

class RegistersPool {
public:
    using Ptr = std::shared_ptr<RegistersPool>;
    using WeakPtr = std::weak_ptr<RegistersPool>;
    static constexpr int anyIdx = -1;

    template<typename TReg>
    class Reg {
        friend class RegistersPool;
    public:
        Reg() {}
        Reg(const RegistersPool::Ptr& regPool) { initialize(regPool); }
        Reg(const RegistersPool::Ptr& regPool, int requestedIdx) { initialize(regPool, requestedIdx); }
        ~Reg() { release(); }
        Reg& operator=(Reg&& other)  noexcept {
            release();
            reg = other.reg;
            regPool = std::move(other.regPool);
            return *this;
        }
        Reg(Reg&& other)  noexcept : reg(other.reg), regPool(std::move(other.regPool)) {}
        operator TReg&() { ensureValid(); return reg; }
        operator const TReg&() const { ensureValid(); return reg; }
        operator Xbyak::RegExp() const { ensureValid(); return reg; }
        int getIdx() const { ensureValid(); return reg.getIdx(); }
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
        bool isInitialized() const { return !regPool.expired(); }

    private:
        void ensureValid() const {
            if (!isInitialized()) {
                OPENVINO_THROW("RegistersPool::Reg is either not initialized or released");
            }
        }

        void initialize(const RegistersPool::Ptr& pool, int requestedIdx = anyIdx) {
            release();
            reg = TReg(pool->template getFree<TReg>(requestedIdx));
            regPool = pool;
        }

    private:
        TReg reg;
        RegistersPool::WeakPtr regPool;
    };

    virtual ~RegistersPool() {
        checkUniqueAndUpdate(false);
    }

    template <ov::runtime::jit::cpu_isa_t isa>
    static Ptr create(std::initializer_list<Xbyak::Reg> regsToExclude);

    static Ptr create(cpu_isa_t isa, std::initializer_list<Xbyak::Reg> regsToExclude);

    template<typename TReg>
    size_t countFree() const {
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            return simdSet.countUnused();
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            return generalSet.countUnused();
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return countUnusedOpmask();
        }
    }

protected:
    class PhysicalSet {
    public:
        PhysicalSet(int size) : isFreeIndexVector(size, true) {}

        void setAsUsed(size_t regIdx) {
            if (regIdx >= isFreeIndexVector.size()) {
                OPENVINO_THROW("regIdx is out of bounds in RegistersPool::PhysicalSet::setAsUsed()");
            }
            if (!isFreeIndexVector[regIdx]) {
                OPENVINO_THROW("Inconsistency in RegistersPool::PhysicalSet::setAsUsed()");
            }
            isFreeIndexVector[regIdx] = false;
        }

        void setAsUnused(size_t regIdx) {
            if (regIdx >= isFreeIndexVector.size()) {
                OPENVINO_THROW("regIdx is out of bounds in RegistersPool::PhysicalSet::setAsUsed()");
            }
            if (isFreeIndexVector[regIdx]) {
                OPENVINO_THROW("Inconsistency in RegistersPool::PhysicalSet::setAsUnused()");
            }
            isFreeIndexVector[regIdx] = true;
        }

        size_t getUnused(size_t requestedIdx) {
            if (requestedIdx == static_cast<size_t>(anyIdx)) {
                return getFirstFreeIndex();
            } else {
                if (requestedIdx >= isFreeIndexVector.size()) {
                    OPENVINO_THROW("requestedIdx is out of bounds in RegistersPool::PhysicalSet::getUnused()");
                }
                if (!isFreeIndexVector[requestedIdx]) {
                    OPENVINO_THROW("The register with index #", requestedIdx, " already used in the RegistersPool");
                }
                return requestedIdx;
            }
        }

        void exclude(Xbyak::Reg reg) {
            isFreeIndexVector.at(reg.getIdx()) = false;
        }

        size_t countUnused() const {
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

    private:
        std::vector<bool> isFreeIndexVector;
    };

    virtual int getFreeOpmask(int requestedIdx) { OPENVINO_THROW("getFreeOpmask: The Opmask is not supported in current instruction set"); }
    virtual void returnOpmaskToPool(int idx) { OPENVINO_THROW("returnOpmaskToPool: The Opmask is not supported in current instruction set"); }
    virtual size_t countUnusedOpmask() const { OPENVINO_THROW("countUnusedOpmask: The Opmask is not supported in current instruction set"); }

    RegistersPool(int simdRegistersNumber)
            : simdSet(simdRegistersNumber) {
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
        for (auto& reg : regsToExclude) {
            if (reg.isXMM() || reg.isYMM() || reg.isZMM()) {
                simdSet.exclude(reg);
            } else if (reg.isREG()) {
                generalSet.exclude(reg);
            }
        }
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
    }

private:
    template<typename TReg>
    int getFree(int requestedIdx) {
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            auto idx = simdSet.getUnused(requestedIdx);
            simdSet.setAsUsed(idx);
            return idx;
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            auto idx = generalSet.getUnused(requestedIdx);
            generalSet.setAsUsed(idx);
            return idx;
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return getFreeOpmask(requestedIdx);
        }
    }

    template<typename TReg>
    void returnToPool(const TReg& reg) {
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            simdSet.setAsUnused(reg.getIdx());
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            generalSet.setAsUnused(reg.getIdx());
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            returnOpmaskToPool(reg.getIdx());
        }
    }

    void checkUniqueAndUpdate(bool isCtor = true) {
        static thread_local bool isCreated = false;
        if (isCtor) {
            if (isCreated) {
                OPENVINO_THROW("There should be only one instance of RegistersPool per thread");
            }
            isCreated = true;
        } else {
            isCreated = false;
        }
    }

    PhysicalSet generalSet {16};
    PhysicalSet simdSet;
};

template <cpu_isa_t isa>
class IsaRegistersPool : public RegistersPool {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude) : RegistersPool(regsToExclude, 32) {}
};

template <>
class IsaRegistersPool<avx512_core> : public RegistersPool {
public:
    IsaRegistersPool() : RegistersPool(32) {
        opmaskSet.exclude(Xbyak::Opmask(0)); // the Opmask(0) has special meaning for some instructions, like gather instruction
    }

    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude)
        : RegistersPool(regsToExclude, 32) {
        for (auto& reg : regsToExclude) {
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

    size_t countUnusedOpmask() const override {
        return opmaskSet.countUnused();
    }

protected:
    PhysicalSet opmaskSet {8};
};

template <>
class IsaRegistersPool<avx512_core_vnni> : public IsaRegistersPool<avx512_core> {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude) : IsaRegistersPool<avx512_core>(regsToExclude) {}
    IsaRegistersPool() : IsaRegistersPool<avx512_core>() {}
};

template <>
class IsaRegistersPool<avx512_core_bf16> : public IsaRegistersPool<avx512_core> {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude) : IsaRegistersPool<avx512_core>(regsToExclude) {}
    IsaRegistersPool() : IsaRegistersPool<avx512_core>() {}
};

template <cpu_isa_t isa>
RegistersPool::Ptr RegistersPool::create(std::initializer_list<Xbyak::Reg> regsToExclude) {
    return std::make_shared<IsaRegistersPool<isa>>(regsToExclude);
}

inline
RegistersPool::Ptr RegistersPool::create(cpu_isa_t isa, std::initializer_list<Xbyak::Reg> regsToExclude) {
#define ISA_SWITCH_CASE(isa) case isa: return std::make_shared<IsaRegistersPool<isa>>(regsToExclude);
    switch (isa) {
        ISA_SWITCH_CASE(sse42)
        ISA_SWITCH_CASE(avx)
        ISA_SWITCH_CASE(avx2)
        ISA_SWITCH_CASE(avx512_core)
        ISA_SWITCH_CASE(avx512_core_vnni)
        ISA_SWITCH_CASE(avx512_core_bf16)
        case avx512_vpopcnt: return std::make_shared<IsaRegistersPool<avx512_core>>(regsToExclude);
        default:
            OPENVINO_THROW("Invalid isa argument in RegistersPool::create(): ", isa);
        }
    OPENVINO_THROW("Invalid isa argument in RegistersPool::create()");
#undef ISA_SWITCH_CASE
}

} // namespace jit
} // namespace runtime
} // namespace ov
