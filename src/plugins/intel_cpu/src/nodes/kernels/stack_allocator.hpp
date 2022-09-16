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
    using Ptr = std::shared_ptr<StackAllocator>;

    class Address;

    StackAllocator(x64::jit_generator& code_gen)
        : StackAllocator{code_gen, code_gen.rbp} {
    }

    StackAllocator(x64::jit_generator& code_gen,
                   const Xbyak::Reg& bp)
        : code_generator{code_gen}
        , base_pointer{bp} {
        checkUnique(true);
        code_gen.mov(base_pointer, code_gen.rsp);
    }

    ~StackAllocator() {
        release();
        checkUnique(false);
    }

    void release() {
        current_offset = {};
        commit();
    }

    void commit() {
        if (current_offset > offset) {
            code_generator.sub(code_generator.rsp, current_offset - offset);
            offset = current_offset;
        } else if (offset > current_offset) {
            code_generator.add(code_generator.rsp, offset - current_offset);
            offset = current_offset;
        }
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
        Xbyak::Address address;
        size_t offset{};
        size_t size{};
    };

    Allocation::Ptr allocate(const size_t alloc_size) {
        std::vector<Allocation::Ptr> free_allocations{};
        for (const auto& alloc : allocations) {
            if (!alloc->is_used && alloc_size <= alloc->size) {
                free_allocations.push_back(alloc);
            }
        }
        if (!free_allocations.empty()) {
            std::sort(free_allocations.begin(), free_allocations.end(),
                [](const Allocation::Ptr& alloc0, const Allocation::Ptr& alloc1) {
                    return alloc0->size < alloc1->size;
                });
            const auto alloc = free_allocations.front();
            alloc->is_used = true;
            return alloc;
        } else {
            current_offset += alloc_size;
            Xbyak::Address addr = code_generator.ptr[base_pointer - current_offset];
            const auto alloc = std::make_shared<Allocation>(addr, current_offset, alloc_size);
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

    x64::jit_generator& code_generator;
    const Xbyak::Reg& base_pointer;

    size_t offset{};
    size_t current_offset{};
    std::vector<Allocation::Ptr> allocations{};
};

class StackAllocator::Address final {
public:
    Address() = default;

    Address(StackAllocator::Ptr stack_allocator,
        const size_t alloc_size)
        : stack_allocator_{stack_allocator}
        , allocation_{stack_allocator_->allocate(alloc_size)} {
    }

    ~Address() {
        release();
    }

    Address(Address&& addr) noexcept {
        this->operator=(std::move(addr));
    }

    Address& operator=(Address&& addr) noexcept {
        release();
        stack_allocator_ = std::move(addr.stack_allocator_);
        allocation_ = std::move(addr.allocation_);
        return *this;
    }

    void release() {
        if (allocation_) {
            allocation_->is_used = false;
        }
        if (stack_allocator_) {
            stack_allocator_->deallocate();
        }
        allocation_ = {};
        stack_allocator_ = {};
    }

    operator Xbyak::Address&() {
        ensureValid();
        stack_allocator_->commit();
        return allocation_->address;
    }

    operator const Xbyak::Address&() const {
        ensureValid();
        stack_allocator_->commit();
        return allocation_->address;
    }

    Address& operator=(const Xbyak::Xmm& vmm) {
        stack_mov(*this, vmm);
        return *this;
    }

    Address& operator=(const Xbyak::Reg& reg) {
        stack_mov(*this, reg);
        return *this;
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
        if (!stack_allocator_ || !allocation_) {
            IE_THROW() << "StackAllocator::Address is either not initialized or released !!";
        }
    }

    x64::jit_generator& generator() const {
        return stack_allocator_->code_generator;
    }

    StackAllocator::Ptr stack_allocator_;
    Allocation::Ptr allocation_;
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
