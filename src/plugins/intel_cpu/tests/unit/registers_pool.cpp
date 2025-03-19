// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "nodes/kernels/x64/registers_pool.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common/nstl.hpp"

using namespace ov::intel_cpu;
using namespace dnnl::impl::cpu;

template <class T>
class RegPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (typename T::RegT(0).isREG()) { // for general purpose registers Reg8, Reg16, Reg32, Reg64
            regNumber = 15; // the RSP register excluded by default
        } else if (typename T::RegT(0).isOPMASK()) {
            regNumber = 8;
        } else { // SIMD registers
            regNumber = x64::cpu_isa_traits<T::IsaParam::isa>::n_vregs;
        }
    }

    int regNumber;
};

TYPED_TEST_SUITE_P(RegPoolTest);

TYPED_TEST_P(RegPoolTest, get_return_by_scope) {
    using XbyakRegT = typename TypeParam::RegT;
    RegistersPool::Ptr regPool = RegistersPool::create<TypeParam::IsaParam::isa>({});
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber);
    {
        RegistersPool::Reg<XbyakRegT> reg{regPool};
        OV_ASSERT_NO_THROW([[maybe_unused]] auto val = static_cast<XbyakRegT &>(reg));
        ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber - 1);
    }
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber);
}

TYPED_TEST_P(RegPoolTest, get_return_by_method) {
    using XbyakRegT = typename TypeParam::RegT;
    RegistersPool::Ptr regPool = RegistersPool::create<TypeParam::IsaParam::isa>({});
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber);
    RegistersPool::Reg<XbyakRegT> reg{regPool};
    OV_ASSERT_NO_THROW([[maybe_unused]] auto val = static_cast<XbyakRegT &>(reg));
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber - 1);
    reg.release();
    ASSERT_ANY_THROW([[maybe_unused]] auto val = static_cast<XbyakRegT &>(reg));
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber);
    reg = RegistersPool::Reg<XbyakRegT>{regPool};
    OV_ASSERT_NO_THROW([[maybe_unused]] auto val = static_cast<XbyakRegT &>(reg));
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber - 1);
}

TYPED_TEST_P(RegPoolTest, default_ctor) {
    using XbyakRegT = typename TypeParam::RegT;
    RegistersPool::Ptr regPool = RegistersPool::create<TypeParam::IsaParam::isa>({});
    RegistersPool::Reg<XbyakRegT> reg;
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber);
    ASSERT_ANY_THROW([[maybe_unused]] auto val = static_cast<XbyakRegT &>(reg));
}

TYPED_TEST_P(RegPoolTest, get_all) {
    using XbyakRegT = typename TypeParam::RegT;
    RegistersPool::Ptr regPool = RegistersPool::create<TypeParam::IsaParam::isa>({});
    using Ptr = std::shared_ptr<RegistersPool::Reg<XbyakRegT>>;
    std::vector<Ptr> regs(this->regNumber);
    for (int c = 0; c < this->regNumber; ++c) {
        regs[c] = std::make_shared<RegistersPool::Reg<XbyakRegT>>(regPool);
    }
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), 0);
    ASSERT_ANY_THROW(RegistersPool::Reg<XbyakRegT>{regPool});
    regs.clear();
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber);
}

TYPED_TEST_P(RegPoolTest, move) {
    using XbyakRegT = typename TypeParam::RegT;
    RegistersPool::Ptr regPool = RegistersPool::create<TypeParam::IsaParam::isa>({});
    RegistersPool::Reg<XbyakRegT> reg{regPool};
    OV_ASSERT_NO_THROW([[maybe_unused]] auto val = static_cast<XbyakRegT &>(reg));
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber - 1);
    RegistersPool::Reg<XbyakRegT> reg2{regPool};
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber - 2);
    auto regIdx = reg.getIdx();
    reg2 = std::move(reg);
    ASSERT_EQ(reg2.getIdx(), regIdx);
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber - 1);
    ASSERT_ANY_THROW([[maybe_unused]] auto val = static_cast<XbyakRegT &>(reg));
    OV_ASSERT_NO_THROW([[maybe_unused]] auto val = static_cast<XbyakRegT &>(reg2));
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), this->regNumber - 1);
}


TYPED_TEST_P(RegPoolTest, fixed_idx) {
    using XbyakRegT = typename TypeParam::RegT;
    RegistersPool::Ptr regPool = RegistersPool::create<TypeParam::IsaParam::isa>({});
    using Ptr = std::shared_ptr<RegistersPool::Reg<XbyakRegT>>;
    std::vector<Ptr> regs(this->regNumber);
    for (int c = 0; c < this->regNumber; ++c) {
        if (c == Xbyak::Operand::RSP) continue;
        regs[c] = std::make_shared<RegistersPool::Reg<XbyakRegT>>(regPool, c);
        ASSERT_EQ(regs[c]->getIdx(), c);
    }
    regs[0]->release();
    ASSERT_ANY_THROW(RegistersPool::Reg<XbyakRegT>(regPool, 1));
    OV_ASSERT_NO_THROW(RegistersPool::Reg<XbyakRegT>(regPool, 0));
}

TYPED_TEST_P(RegPoolTest, exclude) {
    using XbyakRegT = typename TypeParam::RegT;
    static constexpr int excludedIdx = 0;
    RegistersPool::Ptr regPool = RegistersPool::create<TypeParam::IsaParam::isa>({
                                                                                         XbyakRegT(excludedIdx)
                                                                                 });
    using Ptr = std::shared_ptr<RegistersPool::Reg<XbyakRegT>>;
    std::vector<Ptr> regs(this->regNumber - 1);
    std::set<int> idxsInUse;
    for (int c = 0; c < this->regNumber - 1; ++c) {
        regs[c] = std::make_shared<RegistersPool::Reg<XbyakRegT>>(regPool);
        idxsInUse.emplace(regs[c]->getIdx());
    }
    ASSERT_EQ(regPool->countFree<XbyakRegT>(), 0);
    ASSERT_TRUE(idxsInUse.find(excludedIdx) == idxsInUse.end());
}

namespace combiner {

template<class Reg, class Isa>
struct Case {
    using RegT = Reg;
    using IsaParam = Isa;
};

template<class TupleType, class TupleParam, std::size_t I>
struct make_case {
    static constexpr std::size_t N = std::tuple_size<TupleParam>::value;

    using type = Case<typename std::tuple_element<I / N, TupleType>::type,
            typename std::tuple_element<I % N, TupleParam>::type>;
};

template<class T1, class T2, class Is>
struct make_combinations;

template<size_t ...> struct index_sequence  { };
template<size_t N, size_t ...S> struct make_index_sequence_impl : make_index_sequence_impl <N - 1, N - 1, S...> { };
template<size_t ...S> struct make_index_sequence_impl <0, S...> { using type = index_sequence<S...>; };
template<size_t N> using make_index_sequence = typename make_index_sequence_impl<N>::type;

template<class TupleType, class TupleParam, std::size_t... Is>
struct make_combinations<TupleType, TupleParam, index_sequence<Is...>> {
    using tuples = std::tuple<typename make_case<TupleType, TupleParam, Is>::type...>;
};

template<class TupleTypes, class... Params>
using Combinations_t = typename make_combinations
        <TupleTypes,
                std::tuple<Params...>,
                make_index_sequence<(std::tuple_size<TupleTypes>::value) *(sizeof...(Params))>>::tuples;

template<class T>
struct TestTypesCombiner;

template<class ...T>
struct TestTypesCombiner<std::tuple<T...>> {
    using Types = ::testing::Types<T...>;
};

} // namespace combiner

template<x64::cpu_isa_t Isa>
struct IsaParam { static constexpr x64::cpu_isa_t isa = Isa; };

using TestTypes = combiner::TestTypesCombiner<combiner::Combinations_t<
        std::tuple<
                Xbyak::Reg8, Xbyak::Reg16, Xbyak::Reg32, Xbyak::Reg64, Xbyak::Xmm, Xbyak::Ymm, Xbyak::Zmm
        >,
        IsaParam<x64::sse41>,
        IsaParam<x64::avx>,
        IsaParam<x64::avx2>,
        IsaParam<x64::avx2_vnni> >>::Types;

using TestTypesAvx512 = combiner::TestTypesCombiner<combiner::Combinations_t<
        std::tuple<
                Xbyak::Reg8, Xbyak::Reg16, Xbyak::Reg32, Xbyak::Reg64, Xbyak::Xmm, Xbyak::Ymm, Xbyak::Zmm, Xbyak::Opmask
        >,
        IsaParam<x64::avx512_core>,
        IsaParam<x64::avx512_core_vnni>,
        IsaParam<x64::avx512_core_bf16> >>::Types;

REGISTER_TYPED_TEST_SUITE_P(RegPoolTest,
                            get_return_by_scope,
                            get_return_by_method,
                            default_ctor,
                            get_all,
                            move,
                            fixed_idx,
                            exclude);

INSTANTIATE_TYPED_TEST_SUITE_P(testIsaAndRegTypes, RegPoolTest, TestTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(testIsaAndRegTypesAvx512, RegPoolTest, TestTypesAvx512);


const int simdRegNumber = x64::cpu_isa_traits<x64::avx2>::n_vregs;
const int freeGeneralRegNumber = 15;

TEST(RegistersPoolTests, simd_and_general) {
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber);
    ASSERT_EQ(regPool->countFree<Xbyak::Reg64>(), freeGeneralRegNumber);
    {
        RegistersPool::Reg<Xbyak::Xmm> reg{regPool};
        OV_ASSERT_NO_THROW([[maybe_unused]] auto val = static_cast<Xbyak::Xmm &>(reg));
        ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber - 1);
        ASSERT_EQ(regPool->countFree<Xbyak::Ymm>(), simdRegNumber - 1);
        ASSERT_EQ(regPool->countFree<Xbyak::Reg64>(), freeGeneralRegNumber);
    }
    ASSERT_EQ(regPool->countFree<Xbyak::Xmm>(), simdRegNumber);
    ASSERT_EQ(regPool->countFree<Xbyak::Reg64>(), freeGeneralRegNumber);
}

TEST(RegistersPoolTests, second_pool_exception) {
    RegistersPool::Ptr regPool = RegistersPool::create<x64::avx2>({});
    ASSERT_ANY_THROW(RegistersPool::create<x64::avx2>({}));
}

TEST(RegistersPoolTests, get_all) {
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

