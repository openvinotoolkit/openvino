// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/reference/utils/jit_generator.hpp"
namespace ov {
namespace reference {
namespace jit {

class RegistersPool {
public:
    using Ptr = std::shared_ptr<RegistersPool>;
    using WeakPtr = std::weak_ptr<RegistersPool>;
    static constexpr int any_idx = -1;

    template <typename TReg>
    class Reg {
        friend class RegistersPool;

    public:
        Reg() {}
        Reg(const RegistersPool::Ptr& regPool) {
            initialize(regPool);
        }
        Reg(const RegistersPool::Ptr& regPool, int requested_idx) {
            initialize(regPool, requested_idx);
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
        operator TReg&() {
            ensure_valid();
            return reg;
        }
        operator const TReg&() const {
            ensure_valid();
            return reg;
        }
        operator Xbyak::RegExp() const {
            ensure_valid();
            return reg;
        }
        int getIdx() const {
            ensure_valid();
            return reg.getIdx();
        }
        friend Xbyak::RegExp operator+(const Reg& lhs, const Xbyak::RegExp& rhs) {
            lhs.ensure_valid();
            return lhs.operator Xbyak::RegExp() + rhs;
        }
        void release() {
            if (auto pool = regPool.lock()) {
                try {
                    pool->return_to_pool(reg);
                } catch (...) {
                    // This function is called by destructor and should not throw. Well formed Reg object won't cause
                    // any exception throw from return_to_pool, while on badly formed object the destructor is most
                    // likely called during exception stack unwind.
                }
                regPool.reset();
            }
        }
        bool is_initialized() const {
            return !regPool.expired();
        }

    private:
        void ensure_valid() const {
            if (!is_initialized()) {
                OPENVINO_THROW("RegistersPool::Reg is either not initialized or released");
            }
        }

        void initialize(const RegistersPool::Ptr& pool, int requested_idx = any_idx) {
            release();
            reg = TReg(pool->template get_free<TReg>(requested_idx));
            regPool = pool;
        }

    private:
        TReg reg;
        RegistersPool::WeakPtr regPool;
    };

    static thread_local bool is_created;

    virtual ~RegistersPool() {
        is_created = false;
    }

    template <ov::reference::jit::cpu_isa_t isa>
    static Ptr create(std::initializer_list<Xbyak::Reg> regsToExclude);

    static Ptr create(cpu_isa_t isa, std::initializer_list<Xbyak::Reg> regsToExclude);

    template <typename TReg>
    size_t count_free() const {
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            return m_simd_set.count_unused();
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            return m_general_set.count_unused();
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return count_unused_opmask();
        }
    }

protected:
    class PhysicalSet {
    public:
        PhysicalSet(int size) : m_is_free_index_vector(size, true) {}

        void set_as_used(size_t reg_idx);

        void set_as_unused(size_t reg_idx);

        size_t get_unused(size_t requested_idx);

        void exclude(Xbyak::Reg reg) {
            m_is_free_index_vector.at(reg.getIdx()) = false;
        }

        size_t count_unused() const;

    private:
        size_t get_first_free_index();

    private:
        std::vector<bool> m_is_free_index_vector;
    };

    virtual int get_free_opmask(int requested_idx) {
        OPENVINO_THROW("get_free_opmask: The Opmask is not supported in current instruction set");
    }
    virtual void return_opmask_to_pool(int idx) {
        OPENVINO_THROW("return_opmask_to_pool: The Opmask is not supported in current instruction set");
    }
    virtual size_t count_unused_opmask() const {
        OPENVINO_THROW("count_unused_opmask: The Opmask is not supported in current instruction set");
    }

    RegistersPool(int simd_registers_number);

    RegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude, int simd_registers_number);

private:
    template <typename TReg>
    int get_free(int requested_idx) {
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            auto idx = m_simd_set.get_unused(requested_idx);
            m_simd_set.set_as_used(idx);
            return static_cast<int>(idx);
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            auto idx = m_general_set.get_unused(requested_idx);
            m_general_set.set_as_used(idx);
            return static_cast<int>(idx);
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return get_free_opmask(requested_idx);
        }
    }

    template <typename TReg>
    void return_to_pool(const TReg& reg) {
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            m_simd_set.set_as_unused(reg.getIdx());
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            m_general_set.set_as_unused(reg.getIdx());
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return_opmask_to_pool(reg.getIdx());
        }
    }

    void check_unique_and_update();

    PhysicalSet m_general_set;
    PhysicalSet m_simd_set;
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
        m_opmask_set.exclude(
            Xbyak::Opmask(0));  // the Opmask(0) has special meaning for some instructions, like gather instruction
    }

    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude) : RegistersPool(regsToExclude, 32) {
        for (auto& reg : regsToExclude) {
            if (reg.isOPMASK()) {
                m_opmask_set.exclude(reg);
            }
        }
    }

    int get_free_opmask(int requested_idx) override {
        auto idx = static_cast<int>(m_opmask_set.get_unused(requested_idx));
        m_opmask_set.set_as_used(idx);
        return idx;
    }

    void return_opmask_to_pool(int idx) override {
        m_opmask_set.set_as_unused(idx);
    }

    size_t count_unused_opmask() const override {
        return m_opmask_set.count_unused();
    }

protected:
    PhysicalSet m_opmask_set{8};
};

template <cpu_isa_t isa>
RegistersPool::Ptr RegistersPool::create(std::initializer_list<Xbyak::Reg> regsToExclude) {
    return std::make_shared<IsaRegistersPool<isa>>(regsToExclude);
}

inline RegistersPool::Ptr RegistersPool::create(cpu_isa_t isa, std::initializer_list<Xbyak::Reg> regsToExclude) {
#define ISA_SWITCH_CASE(isa) \
    case isa:                \
        return std::make_shared<IsaRegistersPool<isa>>(regsToExclude);
    switch (isa) {
        ISA_SWITCH_CASE(avx2)
        ISA_SWITCH_CASE(avx512_core)
    default:
        OPENVINO_THROW("Invalid isa argument in RegistersPool::create(): ", isa);
    }
#undef ISA_SWITCH_CASE
}

}  // namespace jit
}  // namespace reference
}  // namespace ov
