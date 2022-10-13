// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <ie/ie_common.h>
#include <cpu/x64/jit_generator.hpp>

namespace ov {
namespace intel_cpu {

namespace x64 = dnnl::impl::cpu::x64;

class StackAllocator final {
public:
    class Transaction;
    class Address;
    template<typename TReg>
    class Reg;

    StackAllocator(x64::jit_generator& code_gen)
        : StackAllocator{code_gen, code_gen.rbp} {
    }

    StackAllocator(x64::jit_generator& code_gen, const Xbyak::Reg& bp)
        : StackAllocator{code_gen, bp, 1} {
    }

    StackAllocator(x64::jit_generator& code_gen, const size_t alignment)
        : StackAllocator{code_gen, code_gen.rbp, alignment} {
    }

    StackAllocator(x64::jit_generator& code_gen,
                   const Xbyak::Reg& bp,
                   const size_t alignment)
        : code_generator{code_gen}
        , base_pointer{Xbyak::Reg64{bp.getIdx()}}
        , alignment{alignment} {
        checkUnique(true);
        alignStack(true);
        code_generator.mov(base_pointer, code_generator.rsp);
    }

    ~StackAllocator() {
        release();
        alignStack(false);
        checkUnique(false);
    }

    void release() {
        current_offset = {};
        commit();
    }

    friend void stack_mov(Address& addr, const Xbyak::Xmm& vmm);
    friend void stack_mov(Address& addr, const Xbyak::Reg& reg);
    friend void stack_mov(const Xbyak::Xmm& vmm, const Address& addr);
    friend void stack_mov(const Xbyak::Reg& reg, const Address& addr);

private:
    struct Allocation {
        using Ptr = std::shared_ptr<Allocation>;

        Allocation(const Xbyak::Address& address,
                   const size_t offset,
                   const size_t size)
           : address(address)
           , offset(offset)
           , size(size) {}

        bool is_used = true;
        bool is_transaction = false;
        Xbyak::Address address;
        size_t offset{};
        size_t size{};
    };

    void alignStack(bool isCtor) {
        if (1 != alignment) {
            constexpr size_t kReg64Size = 0x08;
            if (isCtor) {
                Xbyak::Label l_stack_aligned;
                const Xbyak::Reg64 reg_base_stack_offset{base_pointer.getIdx()};
                code_generator.mov(reg_base_stack_offset, static_cast<uint64_t>(kReg64Size));

                const Xbyak::Reg64 reg_base_addr{Xbyak::Operand::RAX};
                const Xbyak::Reg64 reg_reminder{Xbyak::Operand::RDX};
                const Xbyak::Reg64 reg_alignment{Xbyak::Operand::RCX};

                code_generator.push(reg_base_addr);
                code_generator.push(reg_reminder);
                code_generator.push(reg_alignment);

                code_generator.xor_(reg_reminder, reg_reminder);

                code_generator.mov(reg_base_addr, code_generator.rsp);
                code_generator.add(reg_base_addr, 3 * kReg64Size - kReg64Size);
                code_generator.mov(reg_alignment, alignment);
                code_generator.idiv(reg_alignment);
                code_generator.cmp(reg_reminder, static_cast<uint64_t>(0x00));
                code_generator.je(l_stack_aligned);
                code_generator.add(reg_base_stack_offset, reg_reminder);
                code_generator.L(l_stack_aligned);

                code_generator.pop(reg_alignment);
                code_generator.pop(reg_reminder);
                code_generator.pop(reg_base_addr);

                code_generator.sub(code_generator.rsp, reg_base_stack_offset);
                code_generator.mov(code_generator.ptr[code_generator.rsp], reg_base_stack_offset);
            } else {
                code_generator.add(code_generator.rsp, code_generator.ptr[code_generator.rsp]);
            }
        }
    }

    Allocation::Ptr allocate(const size_t alloc_size,
                             const size_t requested_alignment,
                             const bool is_transaction = false) {
        if (alignment % requested_alignment != 0) {
            IE_THROW() << "Requested alignment should have 0 reminder of alignment % align !!";
        }

        std::vector<Allocation::Ptr> free_allocations{};
        for (const auto& alloc : allocations) {
            if (!alloc->is_used &&
                alloc_size <= alloc->size &&
                (alloc->offset % requested_alignment) == 0) {
                free_allocations.push_back(alloc);
            }
        }
        std::sort(free_allocations.begin(), free_allocations.end(),
            [](const Allocation::Ptr& alloc0, const Allocation::Ptr& alloc1) {
                return alloc0->size < alloc1->size;
            });
        if (!free_allocations.empty()) {
            const auto alloc = free_allocations.front();
            alloc->is_used = true;
            alloc->is_transaction = is_transaction;
            return alloc;
        } else {
            size_t alloc_offset = 0;
            if (requested_alignment > 1) {
                alloc_offset = (requested_alignment - ((current_offset+alloc_size) % requested_alignment));
            }

            const size_t aligned_alloc_size = alloc_offset + alloc_size;
            current_offset += aligned_alloc_size;
            Xbyak::Address addr = code_generator.ptr[base_pointer - current_offset];
            const auto alloc = std::make_shared<Allocation>(addr, current_offset, aligned_alloc_size);
            alloc->is_transaction = is_transaction;
            allocations.push_back(alloc);
            return alloc;
        }
    }

    void deallocate() {
        while (!allocations.empty()) {
            const auto& last = allocations.back();
            if (last->is_used) {
                break;
            }
            current_offset -= last->size;
            allocations.pop_back();
        }
    }

    void checkUnique(bool isCtor) {
        static thread_local bool isCreated = false;
        if (isCtor) {
            if (isCreated) {
                IE_THROW() << "There should be only one instance of StackAllocator per thread !!";
            }
            isCreated = true;
        } else {
            isCreated = false;
        }
    }

    bool isTransaction() const {
        return is_transaction_;
    }

    void begin() {
        is_transaction_ = true;
    }

    void commit() {
        if (current_offset > offset) {
            code_generator.sub(code_generator.rsp, current_offset - offset);
            offset = current_offset;
        } else if (offset > current_offset) {
            code_generator.add(code_generator.rsp, offset - current_offset);
            offset = current_offset;
        }
        is_transaction_ = false;
        for (auto& alloc : allocations) {
            alloc->is_transaction = false;
        }
    }

    x64::jit_generator& code_generator;
    const Xbyak::Reg base_pointer;

    bool is_transaction_{};
    size_t offset{};
    size_t current_offset{};
    size_t alignment{};
    std::vector<Allocation::Ptr> allocations{};
};

void stack_mov(StackAllocator::Address& addr, const Xbyak::Xmm& vmm);
void stack_mov(StackAllocator::Address& addr, const Xbyak::Reg& reg);
void stack_mov(const Xbyak::Xmm& vmm, const StackAllocator::Address& addr);
void stack_mov(const Xbyak::Reg& reg, const StackAllocator::Address& addr);

class StackAllocator::Transaction {
public:
    friend class StackAllocator::Address;

    Transaction(StackAllocator& stack_allocator)
        : stack_allocator_{stack_allocator} {
        checkUnique(true);
    }

    ~Transaction() {
        checkUnique(false);
        commit();
    }

    void begin() {
        stack_allocator_.begin();
    }

    void commit() {
        stack_allocator_.commit();
    }

private:
    void checkUnique(bool isCtor) {
        static thread_local bool isCreated = false;
        if (isCtor) {
            if (isCreated) {
                IE_THROW() << "There should be only one instance of Transaction per thread !!";
            }
            isCreated = true;
        } else {
            isCreated = false;
        }
    }

    StackAllocator& stack_allocator_;
};

class StackAllocator::Address {
public:
    Address(Transaction& transaction,
            const size_t alloc_size,
            const size_t requested_alignment = 1)
            : transaction_{&transaction}
            , stack_allocator_{transaction.stack_allocator_}
            , allocation_{stack_allocator_.allocate(alloc_size, requested_alignment, true)} {
        transaction.begin();
    }

    Address(StackAllocator& stack_allocator,
            const size_t alloc_size,
            const size_t requested_alignment = 1)
        : stack_allocator_{stack_allocator}
        , allocation_{stack_allocator_.allocate(alloc_size, requested_alignment)} {
        if (stack_allocator_.isTransaction()) {
            IE_THROW() << "Cannot allocate Address out of transaction. Please, finish first transaction !!";
        }
        stack_allocator_.commit();
    }

    Address(const Address& addr) = delete;
    Address& operator=(const Address& addr) = delete;
    Address(Address&& addr) noexcept = delete;
    Address& operator=(Address&& addr) noexcept = delete;

    virtual ~Address() {
        if (transaction_) {
            release(*transaction_);
        } else {
            release();
            stack_allocator_.commit();
        }
    }

    void release(Transaction& transaction) {
        if (allocation_) {
            transaction.begin();
            allocation_->is_used = false;
            stack_allocator_.deallocate();
        }
        allocation_ = {};
    }

    void release() {
        if (allocation_) {
            if (stack_allocator_.isTransaction()) {
                IE_THROW() << "Cannot release Address out of transaction. Please, finish first transaction !!";
            }
            allocation_->is_used = false;
            stack_allocator_.deallocate();
            stack_allocator_.commit();
        }
        allocation_ = {};
    }

    operator Xbyak::Address&() {
        ensureValid();
        return allocation_->address;
    }

    operator const Xbyak::Address&() const {
        ensureValid();
        return allocation_->address;
    }

    virtual Address& operator=(const Xbyak::Xmm& vmm) {
        stack_mov(*this, vmm);
        return *this;
    }

    virtual Address& operator=(const Xbyak::Reg& reg) {
        stack_mov(*this, reg);
        return *this;
    }

    explicit operator bool() const {
        return isInitialized();
    }

    bool isInitialized() const {
        return allocation_ && !allocation_->is_transaction;
    }

    friend void ::ov::intel_cpu::stack_mov(Address& addr, const Xbyak::Xmm& vmm);
    friend void ::ov::intel_cpu::stack_mov(Address& addr, const Xbyak::Reg& reg);
    friend void ::ov::intel_cpu::stack_mov(const Xbyak::Xmm& vmm, const Address& addr);
    friend void ::ov::intel_cpu::stack_mov(const Xbyak::Reg& reg, const Address& addr);

private:
    void ensureSize(const Xbyak::Reg& reg) const {
        ensureValid();
        const size_t reg_size = reg.getBit() / 8;
        if (reg_size > allocation_->size) {
            IE_THROW() << "reg size is bigger than space allocated in StackAllocator !!";
        }
    }

    void ensureValid() const {
        if (!isInitialized()) {
            IE_THROW() << "StackAllocator::Address is either not initialized or released !!";
        }
    }

    x64::jit_generator& generator() const {
        return stack_allocator_.code_generator;
    }

    Transaction* transaction_{};
    StackAllocator& stack_allocator_;
    Allocation::Ptr allocation_;
};

template<typename TReg>
class StackAllocator::Reg : public StackAllocator::Address {
public:
    static_assert(std::is_base_of<Xbyak::Reg, TReg>::value, "TReg should be a Xbyak::Reg based !!");

    Reg(StackAllocator::Transaction& transaction)
        : Address{transaction, TReg{}.getBit() / 8, getAlignment()} {
    }

    Reg(StackAllocator& stack_allocator)
        : Address{stack_allocator, TReg{}.getBit() / 8, getAlignment()} {
    }

    Reg& operator=(const Xbyak::Xmm& vmm) override {
        Address::operator=(vmm);
        return *this;
    }

    Reg& operator=(const Xbyak::Reg& reg) override {
        Address::operator=(reg);
        return *this;
    }

private:
    static size_t getAlignment() {
        if (std::is_same<TReg, Xbyak::Zmm>::value) {
            return x64::cpu_isa_traits<x64::avx512_core>::vlen;
        } else if (std::is_same<TReg, Xbyak::Ymm>::value) {
            return x64::cpu_isa_traits<x64::avx2>::vlen;
        } else if (std::is_same<TReg, Xbyak::Xmm>::value) {
            return x64::cpu_isa_traits<x64::sse41>::vlen;
        } else {
            return 1;
        }
    }
};

inline
void stack_mov(StackAllocator::Address& addr, const Xbyak::Xmm& vmm) {
    addr.ensureSize(vmm);
    x64::jit_generator& generator = addr.generator();
    if (vmm.isXMM()) {
        generator.uni_vmovdqu(addr.allocation_->address, Xbyak::Xmm{vmm.getIdx()});
    } else if (vmm.isYMM()) {
        generator.uni_vmovdqu(addr.allocation_->address, Xbyak::Ymm{vmm.getIdx()});
    } else if (vmm.isZMM()) {
        generator.uni_vmovdqu(addr.allocation_->address, Xbyak::Zmm{vmm.getIdx()});
    } else {
        IE_THROW() << "Unknown simd register !!";
    }
}

inline
void stack_mov(StackAllocator::Address& addr, const Xbyak::Reg& reg) {
    addr.ensureSize(reg);
    x64::jit_generator& generator = addr.generator();
    if (reg.isREG(8)) {
        generator.mov(addr.allocation_->address, Xbyak::Reg8{reg.getIdx()});
    } else if (reg.isREG(16)) {
        generator.mov(addr.allocation_->address, Xbyak::Reg16{reg.getIdx()});
    } else if (reg.isREG(32)) {
        generator.mov(addr.allocation_->address, Xbyak::Reg32{reg.getIdx()});
    } else if (reg.isREG(64)) {
        generator.mov(addr.allocation_->address, Xbyak::Reg64{reg.getIdx()});
    } else {
        IE_THROW() << "Unknown general purpose register !!";
    }
}

inline
void stack_mov(const Xbyak::Xmm& vmm, const StackAllocator::Address& addr) {
    addr.ensureSize(vmm);
    x64::jit_generator& generator = addr.generator();
    if (vmm.isXMM()) {
        generator.uni_vmovdqu(Xbyak::Xmm{vmm.getIdx()}, addr.allocation_->address);
    } else if (vmm.isYMM()) {
        generator.uni_vmovdqu(Xbyak::Ymm{vmm.getIdx()}, addr.allocation_->address);
    } else if (vmm.isZMM()) {
        generator.uni_vmovdqu(Xbyak::Zmm{vmm.getIdx()}, addr.allocation_->address);
    } else {
        IE_THROW() << "Unknown simd register !!";
    }
}

inline
void stack_mov(const Xbyak::Reg& reg, const StackAllocator::Address& addr) {
    addr.ensureSize(reg);
    x64::jit_generator& generator = addr.generator();
    if (reg.isREG(8)) {
        generator.mov(Xbyak::Reg8{reg.getIdx()}, addr.allocation_->address);
    } else if (reg.isREG(16)) {
        generator.mov(Xbyak::Reg16{reg.getIdx()}, addr.allocation_->address);
    } else if (reg.isREG(32)) {
        generator.mov(Xbyak::Reg32{reg.getIdx()}, addr.allocation_->address);
    } else if (reg.isREG(64)) {
        generator.mov(Xbyak::Reg64{reg.getIdx()}, addr.allocation_->address);
    } else {
        IE_THROW() << "Unknown general purpose register !!";
    }
}

} // namespace intel_cpu
} // namespace ov
