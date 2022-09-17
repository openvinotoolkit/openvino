// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include <cpu/x64/jit_generator.hpp>
#include <nodes/kernels/stack_allocator.hpp>

#include <cpu_memory.h>

using namespace ov::intel_cpu;
using namespace InferenceEngine;

class StackAllocatorTest : public ::testing::Test, public x64::jit_generator {
protected:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(StackAllocatorTest)

    void SetUp() override {
    }

    void TearDown() override {
    }

    template<typename F>
    F create_kernel() {
        const status_t code = jit_generator::create_kernel();
        if (code != dnnl::impl::status::success) {
            IE_THROW() << "Could not create kernel. Error code: " << std::to_string(code) << ". "
                       << "Xbyak error code: " << Xbyak::ConvertErrorToString(Xbyak::GetError());
        }
        return reinterpret_cast<F>(jit_ker());
    }

    void generate() override {
        this->preamble();
        stack_allocator_ = std::make_shared<StackAllocator>(*this);
        kernel_();
        stack_allocator_->commit();
        stack_allocator_.reset();
        this->postamble();
    }

    std::function<void()> kernel_;
    std::shared_ptr<StackAllocator> stack_allocator_;
};

TEST_F(StackAllocatorTest, ValueEqual) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Address reg_100_addr{stack_allocator_, sizeof(int32_t)};
        mov(rbx.cvt32(), 100);
        stack_mov(reg_100_addr, rbx.cvt32());
        mov(rax, 1);
        cmp(rbx.cvt32(), reg_100_addr);
        je(l_equal);
        mov(rax, 0);
        L(l_equal);
    };
    auto f = create_kernel<int(*) ()>();
    int r = f();
    EXPECT_EQ(r, 1);
}

TEST_F(StackAllocatorTest, ValueNotEqual) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Address reg_100_addr{stack_allocator_, sizeof(int32_t)};
        mov(rbx.cvt32(), 100);
        stack_mov(reg_100_addr, rbx.cvt32());
        mov(rcx.cvt32(), reg_100_addr);
        mov(rbx.cvt32(), 201);
        mov(rax, 1);
        cmp(rbx.cvt32(), rcx.cvt32());
        je(l_equal);
        mov(rax, 0);
        L(l_equal);
    };
    auto f = create_kernel<int(*) ()>();
    int r = f();
    EXPECT_EQ(r, 0);
}

TEST_F(StackAllocatorTest, AddressCheck) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Address reg_0_5_addr{stack_allocator_, sizeof(int32_t)};
        EXPECT_EQ(static_cast<Xbyak::Address>(reg_0_5_addr), ptr[rbp - sizeof(int32_t)]);
    };
    create_kernel<int(*) ()>();
}

TEST_F(StackAllocatorTest, LoopSuccess) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        Xbyak::Label l_loop;
        Xbyak::Label l_end;
        xor_(rcx, rcx);
        StackAllocator::Address reg_100_addr{stack_allocator_, sizeof(int32_t)};
        stack_allocator_->commit();
        L(l_loop);
        {
            cmp(rcx, 10);
            je(l_end);
            mov(rbx.cvt32(), 100);
            stack_mov(reg_100_addr, rbx.cvt32());
            mov(rdx.cvt32(), reg_100_addr);
            mov(rbx.cvt32(), 201);
            mov(rax, 1);
            add(rcx, 1);
            cmp(rbx.cvt32(), rdx.cvt32());
            jne(l_equal);
            {
                mov(rax, 0);
                jmp(l_loop);
            }
            L(l_equal);
            {
                jmp(l_loop);
            }
        }
        L(l_end);
        reg_100_addr.release();
    };
    auto f = create_kernel<int(*) ()>();
    const int r = f();
    EXPECT_EQ(r, 1);
}

TEST_F(StackAllocatorTest, LoopFailed) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        Xbyak::Label l_loop;
        Xbyak::Label l_end;
        xor_(rcx, rcx);

        StackAllocator::Address reg_temp0_addr{stack_allocator_, sizeof(float)};
        StackAllocator::Address reg_temp1_addr{stack_allocator_, sizeof(int32_t)};
        StackAllocator::Address reg_100_addr{stack_allocator_, sizeof(int32_t)};
        StackAllocator::Address reg_200_addr{stack_allocator_, sizeof(int32_t)};
        stack_allocator_->commit();
        L(l_loop);
        {
            cmp(rcx, 10);
            je(l_end);
            mov(rbx.cvt32(), 100);
            stack_mov(reg_100_addr, rbx.cvt32());
            mov(rbx.cvt32(), 200);
            reg_200_addr.release();
            // NOTE: During implicit conversion to Xbyak::Address& will be thrown the exception
            stack_mov(reg_200_addr, rbx.cvt32());
            mov(rdx.cvt32(), reg_100_addr);
            mov(rbx.cvt32(), 201);
            cmp(rbx.cvt32(), rdx.cvt32());
            mov(rax, 1);
            add(rcx, 1);
            jne(l_equal);
            {
                reg_temp1_addr.release();
                mov(rax, 0);
                jmp(l_loop);
            }
            L(l_equal);
            {
                jmp(l_loop);
            }
        }
        L(l_end);
    };
    EXPECT_ANY_THROW(create_kernel<int(*) ()>());
}
