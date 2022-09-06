// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "nodes/kernels/registers_pool.hpp"

using namespace ov::intel_cpu;
const int simdRegNumber = x64::cpu_isa_traits<x64::avx2>::n_vregs;
const int freeGeneralRegNumber = 15;

TEST(RegistersPoolTests, get_return_by_scope) {
    if (!x64::mayiuse(x64::avx2)) { return; }
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber);
    ASSERT_EQ(regPool->countFree<Xbyak::Reg64>(), freeGeneralRegNumber);
    {
        RegistersPool::Reg<Xbyak::Xmm> reg{regPool};
        ASSERT_NO_THROW(static_cast<Xbyak::Xmm &>(reg));
        ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber - 1);
        ASSERT_EQ(regPool->countFree<Xbyak::Ymm>(), simdRegNumber - 1);
        ASSERT_EQ(regPool->countFree<Xbyak::Reg64>(), freeGeneralRegNumber);
    }
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber);
}

TEST(RegistersPoolTests, get_return_by_method) {
    if (!x64::mayiuse(x64::avx2)) { return; }
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber);
    RegistersPool::Reg<Xbyak::Xmm> reg{regPool};
    ASSERT_NO_THROW(static_cast<Xbyak::Xmm &>(reg));
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber - 1);
    reg.release();
    ASSERT_ANY_THROW(static_cast<Xbyak::Xmm &>(reg));
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber);
    reg = RegistersPool::Reg<Xbyak::Xmm>{regPool};
    ASSERT_NO_THROW(static_cast<Xbyak::Xmm &>(reg));
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber - 1);
}

TEST(RegistersPoolTests, second_pool_exception) {
    if (!x64::mayiuse(x64::avx2)) { return; }
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    ASSERT_ANY_THROW(RegistersPool::create<x64::avx2>({}));
}

TEST(RegistersPoolTests, default_ctor) {
    if (!x64::mayiuse(x64::avx2)) { return; }
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    RegistersPool::Reg<Xbyak::Xmm> reg;
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber);
    ASSERT_ANY_THROW(static_cast<Xbyak::Xmm &>(reg));
}

TEST(RegistersPoolTests, get_all) {
    if (!x64::mayiuse(x64::avx2)) { return; }
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    using Ptr = std::shared_ptr<RegistersPool::Reg<Xbyak::Xmm>>;
    std::vector<Ptr> regs(simdRegNumber);
    std::vector<int> idxs(simdRegNumber);
    for (int c = 0; c < simdRegNumber; ++c) {
        regs[c] = std::make_shared<RegistersPool::Reg<Xbyak::Xmm>>(regPool);
        idxs[c] = regs[c]->getIdx();
    }
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), 0);
    ASSERT_ANY_THROW(RegistersPool::Reg<Xbyak::Xmm>{regPool});
    std::sort(idxs.begin(), idxs.end());
    for (int c = 0; c < simdRegNumber; ++c) {
        ASSERT_EQ(c, idxs[c]);
    }
    regs.clear();
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber);
}

TEST(RegistersPoolTests, move) {
    if (!x64::mayiuse(x64::avx2)) { return; }
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    RegistersPool::Reg<Xbyak::Xmm> reg{regPool};
    ASSERT_NO_THROW(static_cast<Xbyak::Xmm &>(reg));
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber - 1);
    RegistersPool::Reg<Xbyak::Xmm> reg2 = std::move(reg);
    ASSERT_ANY_THROW(static_cast<Xbyak::Xmm &>(reg));
    ASSERT_NO_THROW(static_cast<Xbyak::Xmm &>(reg2));
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber - 1);
}


TEST(RegistersPoolTests, fixed_idx) {
    if (!x64::mayiuse(x64::avx2)) { return; }
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    using Ptr = std::shared_ptr<RegistersPool::Reg<Xbyak::Xmm>>;
    std::vector<Ptr> regs(simdRegNumber);
    for (int c = 0; c < simdRegNumber; ++c) {
        regs[c] = std::make_shared<RegistersPool::Reg<Xbyak::Xmm>>(regPool, c);
        ASSERT_EQ(regs[c]->getIdx(), c);
    }
    regs[0]->release();
    ASSERT_ANY_THROW(RegistersPool::Reg<Xbyak::Xmm>(regPool, 1));
    ASSERT_NO_THROW(RegistersPool::Reg<Xbyak::Xmm>(regPool, 0));
}

TEST(RegistersPoolTests, exclude) {
    if (!x64::mayiuse(x64::avx2)) { return; }
    static constexpr int excludedIdx = 0;
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({
        Xbyak::Xmm(excludedIdx)
    });
    using Ptr = std::shared_ptr<RegistersPool::Reg<Xbyak::Xmm>>;
    std::vector<Ptr> regs(simdRegNumber - 1);
    std::set<int> idxsInUse;
    for (int c = 0; c < simdRegNumber - 1; ++c) {
        regs[c] = std::make_shared<RegistersPool::Reg<Xbyak::Xmm>>(regPool);
        idxsInUse.emplace(regs[c]->getIdx());
    }
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), 0);
    ASSERT_TRUE(idxsInUse.find(excludedIdx) == idxsInUse.end());
}