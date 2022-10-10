// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <utility>
#include <cpu/x64/jit_generator.hpp>
#include <nodes/kernels/stack_allocator.hpp>

#include <cpu_memory.h>

using namespace ov::intel_cpu;

constexpr int x64::cpu_isa_traits<x64::avx512_core>::vlen;
constexpr int x64::cpu_isa_traits<x64::avx>::vlen;
constexpr int x64::cpu_isa_traits<x64::sse41>::vlen;

class StackAllocatorTest : public ::testing::Test, public x64::jit_generator {
protected:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(StackAllocatorTest)

    StackAllocatorTest()
        : StackAllocatorTest(x64::isa_all) {
    }

    explicit StackAllocatorTest(x64::cpu_isa_t max_cpu_isa)
        : x64::jit_generator(jit_name(), nullptr, 256 * 1024, true, max_cpu_isa) {
    }

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
        stack_allocator_->release();
        stack_allocator_.reset();
        this->postamble();
    }

    std::function<void()> kernel_;
    std::shared_ptr<StackAllocator> stack_allocator_;
};

TEST_F(StackAllocatorTest, Address_ValueEqual) {
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

TEST_F(StackAllocatorTest, Reg32_ValueEqual) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Reg<Xbyak::Reg32> reg_100_addr{stack_allocator_};
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

TEST_F(StackAllocatorTest, Address_ValueNotEqual) {
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

TEST_F(StackAllocatorTest, Reg32_ValueNotEqual) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Reg<Xbyak::Reg32> reg_100_addr{stack_allocator_};
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

TEST_F(StackAllocatorTest, Address_Check) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Address dword_addr{stack_allocator_, sizeof(int32_t)};
        EXPECT_EQ(static_cast<Xbyak::Address>(dword_addr), ptr[rbp - sizeof(int32_t)]);
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

        StackAllocator::Transaction transaction{stack_allocator_};
        StackAllocator::Address reg_temp0_addr{transaction, sizeof(float)};
        StackAllocator::Address reg_temp1_addr{transaction, sizeof(int32_t)};
        StackAllocator::Address reg_100_addr{transaction, sizeof(int32_t)};
        StackAllocator::Address reg_200_addr{transaction, sizeof(int32_t)};
        transaction.commit();
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

TEST_F(StackAllocatorTest, Address_CheckAlignmentFailed) {
    kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Transaction transaction{stack_allocator_};
        StackAllocator::Address byte0_addr{transaction, sizeof(int8_t)};
        StackAllocator::Address xmm0_addr{transaction, 16};
        transaction.commit();
        addps(xmm0, xmm0_addr);
    };
    auto f = create_kernel<int(*) ()>();
    ASSERT_DEATH({
        f();
    }, "");
}

template <typename T>
class AlignedStackAllocatorTest : public StackAllocatorTest {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(AlignedStackAllocatorTest)

    AlignedStackAllocatorTest()
            : StackAllocatorTest(T::isa) {
    }

    void SetUp() override {
    }

    void TearDown() override {
    }

    void generate() override {
        this->preamble();
        stack_allocator_ = std::make_shared<StackAllocator>(*this, x64::cpu_isa_traits<T::isa>::vlen);
        kernel_();
        stack_allocator_->release();
        stack_allocator_.reset();
        this->postamble();
    }
};

template<x64::cpu_isa_t Isa>
struct IsaParam { static constexpr x64::cpu_isa_t isa = Isa; };

using IsaParamTypes = ::testing::Types<IsaParam<x64::sse41>, IsaParam<x64::avx2>, IsaParam<x64::avx512_core>>;
TYPED_TEST_SUITE(AlignedStackAllocatorTest, IsaParamTypes);

TYPED_TEST(AlignedStackAllocatorTest, Address_CheckAlignmentSuccess) {
    this->kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Transaction transaction{this->stack_allocator_};
        StackAllocator::Address byte0_addr{transaction, sizeof(int8_t)};
        StackAllocator::Address xmm0_addr{transaction, 16, 16};
        transaction.commit();
        this->addps(this->xmm0, xmm0_addr);
    };
    auto f = this->template create_kernel<int(*) ()>();
    f();
}

TYPED_TEST(AlignedStackAllocatorTest, Address_CheckAlignmentRegSuccess) {
    this->kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Address byte0_addr{this->stack_allocator_, sizeof(int8_t)};
        StackAllocator::Reg<Xbyak::Xmm> xmm0_addr{this->stack_allocator_};
        this->addps(this->xmm0, xmm0_addr);
    };
    auto f = this->template create_kernel<int(*) ()>();
    f();
}

TYPED_TEST(AlignedStackAllocatorTest, Address_CheckAlignmentReuseAddressSuccess) {
    this->kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Transaction transaction{this->stack_allocator_};
        StackAllocator::Address byte0_addr{transaction, sizeof(int8_t)};
        StackAllocator::Reg<Xbyak::Xmm> xmm0_addr{transaction};
        StackAllocator::Address byte1_addr{transaction, sizeof(int8_t)};
        transaction.commit();
        this->addps(this->xmm0, xmm0_addr);
        xmm0_addr.release();
        StackAllocator::Reg<Xbyak::Xmm> xmm1_addr{transaction};
        transaction.commit();
        this->addps(this->xmm0, xmm1_addr);
    };
    auto f = this->template create_kernel<int(*) ()>();
    f();
}

TYPED_TEST(AlignedStackAllocatorTest, Address_CheckAlignmentReuseAddressSuccess2) {
    this->kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Transaction transaction{this->stack_allocator_};
        StackAllocator::Address byte0_addr{transaction, sizeof(int8_t)};
        StackAllocator::Reg<Xbyak::Xmm> xmm0_addr{transaction};
        StackAllocator::Address byte1_addr{transaction, sizeof(int8_t)};
        StackAllocator::Reg<Xbyak::Xmm> xmm1_addr{this->stack_allocator_};
        transaction.commit();
        this->addps(this->xmm0, xmm0_addr);
        xmm0_addr.release();
        StackAllocator::Reg<Xbyak::Xmm> xmm2_addr{transaction};
        transaction.commit();
        this->addps(this->xmm0, xmm2_addr);
    };

    EXPECT_ANY_THROW(this->template create_kernel<int(*) ()>());
}

TYPED_TEST(AlignedStackAllocatorTest, Address_CheckAlignmentReuseAddressSuccess3) {
    this->kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Transaction transaction{this->stack_allocator_};
        StackAllocator::Address byte0_addr{transaction, sizeof(int8_t)};
        StackAllocator::Reg<Xbyak::Xmm> xmm0_addr{transaction};
        StackAllocator::Address byte1_addr{transaction, sizeof(int8_t)};
        this->addps(this->xmm0, xmm0_addr);
        transaction.commit();
        xmm0_addr.release();
        StackAllocator::Reg<Xbyak::Xmm> xmm1_addr{transaction};
        transaction.commit();
        this->addps(this->xmm0, xmm1_addr);
    };

    EXPECT_ANY_THROW(this->template create_kernel<int(*) ()>());
}

TYPED_TEST(AlignedStackAllocatorTest, Address_CheckAlignmentReuseAddressSuccess4) {
    this->kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Transaction transaction{this->stack_allocator_};
        StackAllocator::Address byte0_addr{transaction, sizeof(int8_t)};
        StackAllocator::Reg<Xbyak::Xmm> xmm0_addr{transaction};
        StackAllocator::Address byte1_addr{transaction, sizeof(int8_t)};
        transaction.commit();
        this->addps(this->xmm0, xmm0_addr);
        xmm0_addr.release(transaction);
        byte1_addr.release(transaction);
        StackAllocator::Reg<Xbyak::Xmm> xmm1_addr{transaction};
        transaction.commit();
        this->addps(this->xmm0, xmm1_addr);
    };

    auto f = this->template create_kernel<int(*) ()>();
    f();
}

TYPED_TEST(AlignedStackAllocatorTest, Address_CheckAlignmentReuseAddressSuccess5) {
    this->kernel_ = [this]() {
        Xbyak::Label l_equal;
        StackAllocator::Transaction transaction{this->stack_allocator_};
        StackAllocator::Address byte0_addr{transaction, sizeof(int8_t)};
        StackAllocator::Reg<Xbyak::Xmm> xmm0_addr{transaction};
        StackAllocator::Address byte1_addr{transaction, sizeof(int8_t)};
        transaction.commit();
        this->uni_vaddps(this->xmm0, this->xmm0, xmm0_addr);
        xmm0_addr.release(transaction);
        byte1_addr.release();
        StackAllocator::Reg<Xbyak::Xmm> xmm1_addr{transaction};
        transaction.commit();
        this->addps(this->xmm0, xmm1_addr);
    };

    EXPECT_ANY_THROW(this->template create_kernel<int(*) ()>());
}

TYPED_TEST(AlignedStackAllocatorTest, Xmm_ValueEqual) {
    static const uint32_t data[4] = {1024, 2135, 3246, 4357};
    this->kernel_ = [this]() {
        Xbyak::Label l_not_equal;
        Xbyak::Xmm vmm0{0};
        Xbyak::Xmm vmm1{1};
        StackAllocator::Reg<Xbyak::Xmm> value_on_stack{this->stack_allocator_};
        this->mov(this->rbx, reinterpret_cast<uintptr_t>(data));
        this->uni_vmovups(vmm0, this->ptr[this->rbx]);
        value_on_stack = vmm0;
        this->uni_vpxor(vmm1, vmm1, vmm1);
        this->uni_vmovups(vmm1, value_on_stack);
        this->mov(this->rax, 0);
        this->uni_vpcmpeqd(vmm0, vmm0, vmm1);
        this->uni_vtestps(vmm0, vmm0);
        this->jz(l_not_equal);
        this->mov(this->rax, 1);
        this->L(l_not_equal);
    };
    auto f = this->template create_kernel<int(*) ()>();
    int r = f();
    EXPECT_EQ(r, 1);
}

TYPED_TEST(AlignedStackAllocatorTest, Xmm_ValueNotEqual) {
    static const uint32_t data[4] = {1024, 2135, 3246, 4357};
    this->kernel_ = [this]() {
        Xbyak::Label l_not_equal;
        Xbyak::Xmm vmm0{0};
        Xbyak::Xmm vmm1{1};
        StackAllocator::Reg<Xbyak::Xmm> value_on_stack{this->stack_allocator_};
        this->mov(this->rbx, reinterpret_cast<uintptr_t>(data));
        this->uni_vmovups(vmm0, this->ptr[this->rbx]);
        value_on_stack = vmm0;
        this->uni_vpxor(vmm1, vmm1, vmm1);
        this->uni_vmovups(vmm1, value_on_stack);
        this->uni_vpxor(vmm0, vmm0, vmm0);
        this->mov(this->rax, 0);
        this->uni_vpcmpeqd(vmm0, vmm0, vmm1);
        this->uni_vtestps(vmm0, vmm0);
        this->jz(l_not_equal);
        this->mov(this->rax, 1);
        this->L(l_not_equal);
    };
    auto f = this->template create_kernel<int(*) ()>();
    int r = f();
    EXPECT_EQ(r, 0);
}

TYPED_TEST(AlignedStackAllocatorTest, Ymm_ValueEqual) {
    if (TypeParam::isa != x64::avx2) {
        GTEST_SKIP() << "Skipping test for isa = " << static_cast<int>(TypeParam::isa);
    }
    static const uint32_t data[8] = {1024, 2135, 3246, 4357,
                                     2124, 3235, 4346, 5457};
    this->kernel_ = [this]() {
        Xbyak::Label l_not_equal;
        Xbyak::Ymm vmm0{0};
        Xbyak::Ymm vmm1{1};
        StackAllocator::Reg<Xbyak::Ymm> value_on_stack{this->stack_allocator_};
        this->mov(this->rbx, reinterpret_cast<uintptr_t>(data));
        this->uni_vmovups(vmm0, this->ptr[this->rbx]);
        value_on_stack = vmm0;
        this->uni_vpxor(vmm1, vmm1, vmm1);
        this->uni_vmovups(vmm1, value_on_stack);
        this->mov(this->rax, 0);
        this->uni_vpcmpeqd(vmm0, vmm0, vmm1);
        this->uni_vtestps(vmm0, vmm0);
        this->jz(l_not_equal);
        this->mov(this->rax, 1);
        this->L(l_not_equal);
    };
    auto f = this->template create_kernel<int(*) ()>();
    int r = f();
    EXPECT_EQ(r, 1);
}

TYPED_TEST(AlignedStackAllocatorTest, Ymm_ValueNotEqual) {
    if (TypeParam::isa != x64::avx2) {
        GTEST_SKIP() << "Skipping test for isa = " << static_cast<int>(TypeParam::isa);
    }
    static const uint32_t data[8] = {1024, 2135, 3246, 4357,
                                     2124, 3235, 4346, 5457};
    this->kernel_ = [this]() {
        Xbyak::Label l_not_equal;
        Xbyak::Ymm vmm0{0};
        Xbyak::Ymm vmm1{1};
        StackAllocator::Reg<Xbyak::Ymm> value_on_stack{this->stack_allocator_};
        this->mov(this->rbx, reinterpret_cast<uintptr_t>(data));
        this->uni_vmovups(vmm0, this->ptr[this->rbx]);
        value_on_stack = vmm0;
        this->uni_vpxor(vmm1, vmm1, vmm1);
        this->uni_vmovups(vmm1, value_on_stack);
        this->uni_vpxor(vmm0, vmm0, vmm0);
        this->mov(this->rax, 0);
        this->uni_vpcmpeqd(vmm0, vmm0, vmm1);
        this->uni_vtestps(vmm0, vmm0);
        this->jz(l_not_equal);
        this->mov(this->rax, 1);
        this->L(l_not_equal);
    };
    auto f = this->template create_kernel<int(*) ()>();
    int r = f();
    EXPECT_EQ(r, 0);
}
