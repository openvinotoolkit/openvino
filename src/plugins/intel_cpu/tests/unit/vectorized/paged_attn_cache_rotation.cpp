// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest/internal/gtest-param-util.h>

#include <memory>
#include <random>
#include <string>

// the includes in the block below are necessary in order for the common.hpp header to be
// instantiated correctly
#include <cstring>
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif
#include "kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/cache_rotation.hpp"
#include "perf_count.h"
#include "utils/plain_tensor.hpp"

using namespace ov::intel_cpu;

template <class T>
using Rank2Matrix = std::vector<std::vector<T>>;

template <class T>
using Rank3Matrix = std::vector<std::vector<std::vector<T>>>;

// Expected layout: [block_size, embedding_size]
template <class T>
std::vector<T> get_block_memory(size_t block_size, size_t embedding_size, const Rank2Matrix<T>& init_values) {
    auto mem = std::vector<T>(block_size * embedding_size);
    if (!init_values.empty()) {
        assert(init_values.size() == block_size);
        assert(init_values[0].size() == embedding_size);
        for (size_t i = 0; i < block_size; i++) {
            for (size_t j = 0; j < embedding_size; j++) {
                mem[i * embedding_size + j] = init_values[i][j];
            }
        }
    }
    return mem;
}

// Expected layout: [num_heads, block_size, embedding_size]
template <class T>
std::vector<T> get_block_memory(size_t num_heads,
                                size_t block_size,
                                size_t embedding_size,
                                const Rank3Matrix<T>& init_values) {
    auto mem = std::vector<T>(num_heads * block_size * embedding_size);
    if (!init_values.empty()) {
        assert(init_values.size() == num_heads);
        assert(init_values[0].size() == block_size);
        assert(init_values[0][0].size() == embedding_size);
        for (size_t i = 0; i < num_heads; i++) {
            for (size_t j = 0; j < block_size; j++) {
                for (size_t k = 0; k < embedding_size; k++) {
                    mem[i * embedding_size * block_size + j * embedding_size + k] = init_values[i][j][k];
                }
            }
        }
    }
    return mem;
}

template <class T>
Rank3Matrix<T> get_matrix_from_mem(std::vector<T> mem_vec,
                                   size_t num_heads,
                                   size_t block_size,
                                   size_t embedding_size) {
    Rank3Matrix<T> retval(num_heads);
    for (size_t i = 0; i < num_heads; i++) {
        retval[i].resize(block_size);
        for (size_t j = 0; j < block_size; j++) {
            retval[i][j].resize(embedding_size);
        }
    }
    for (size_t i = 0; i < num_heads; i++) {
        for (size_t j = 0; j < block_size; j++) {
            for (size_t k = 0; k < embedding_size; k++) {
                retval[i][j][k] = mem_vec[block_size * embedding_size * i + embedding_size * j + k];
            }
        }
    }
    return retval;
}

template <class T>
void compare_with_tolerance(const Rank3Matrix<T>& test_data, const Rank3Matrix<T>& ref_data, T abs_err) {
    ASSERT_EQ(test_data.size(), ref_data.size());
    ASSERT_GT(test_data.size(), 0);

    ASSERT_EQ(test_data[0].size(), ref_data[0].size());
    ASSERT_GT(test_data[0].size(), 0);

    ASSERT_EQ(test_data[0][0].size(), ref_data[0][0].size());
    ASSERT_GT(test_data[0][0].size(), 0);

    for (size_t i = 0; i < test_data.size(); i++) {
        for (size_t j = 0; j < test_data[0].size(); j++) {
            for (size_t k = 0; k < test_data[0][0].size(); k++) {
                T diff = test_data[i][j][k] - ref_data[i][j][k];
                if ((diff > abs_err) || (diff < -abs_err)) {
                    ADD_FAILURE() << std::setprecision(8) << "diff " << diff << " exceeding atol " << abs_err
                                  << " at idx [" << i << ";" << j << ";" << k << "] --- test " << test_data[i][j][k]
                                  << ", ref " << ref_data[i][j][k];
                }
            }
        }
    }
}

template <class T>
static T get_tolerance() {
    return T{};
}

template <>
float get_tolerance<float>() {
    return 1e-6f;
}

template <>
ov::float16 get_tolerance<ov::float16>() {
    return ov::float16{5e-3};
}

template <>
ov::bfloat16 get_tolerance<ov::bfloat16>() {
    return ov::bfloat16{4e-2};
}

template <class TypeParam>
class CacheRotationKernelInputTypeParameterizedTest : public ::testing::Test {
public:
    void SetUp() override {
        Rank3Matrix<TypeParam> values_before_rotation = {
            {
                {1.0f, 1.0f, 1.0f, 1.0f},
                {1.0f, 1.0f, 1.0f, 1.0f},
                {1.0f, 1.0f, 1.0f, 1.0f},
                {1.0f, 1.0f, 1.0f, 1.0f},
            },
            {
                {-2.0f, -2.0f, -2.0f, -2.0f},
                {2.0f, 2.0f, 2.0f, 2.0f},
                {-1.0f, 2.0f, -3.0f, 4.0f},
                {2.0f, 2.0f, 2.0f, 2.0f},
            },
        };
        cache_mem = get_block_memory(num_heads, block_size, embedding_size, values_before_rotation);

        Rank2Matrix<float> rotation_values = {
            {0.5f, 0.70710678f, 0.86602540f, -0.70710678f},
            {0.86602540f, 1.0f, 0.5f, 0.0f},
            {-0.70710678f, 0.0f, 0.70710678f, 1.0f},
            {0.0f, 0.6f, -1.0f, -0.8f},
        };

        rotation_coefficients_mem = get_block_memory(block_size, embedding_size, rotation_values);
    }
    size_t num_heads = 2;
    size_t block_size = 4;
    size_t embedding_size = 4;
    std::vector<TypeParam> cache_mem;
    std::vector<float> rotation_coefficients_mem;
    Rank3Matrix<TypeParam> ref_values_after_rotation = {
        {
            {-0.36602540f, 1.41421356f, 1.36602540f, 0.00000000f},
            {0.36602540f, 1.00000000f, 1.36602540f, 1.00000000f},
            {-1.41421356f, -1.00000000f, 0.00000000f, 1.00000000f},
            {1.00000000f, 1.40000000f, -1.00000000f, -0.20000000f},
        },
        {
            {0.73205081f, -2.82842712f, -2.73205081f, 0.00000000f},
            {0.73205081f, 2.00000000f, 2.73205081f, 2.00000000f},
            {2.82842712f, -4.00000000f, 1.41421356f, 2.00000000f},
            {2.00000000f, 2.80000000f, -2.00000000f, -0.40000000f},
        },
    };

    void test_block_opt_vs_ref(size_t num_heads, size_t embedding_size, size_t block_size) {
        auto cache_block_mem_ref = get_block_memory(num_heads, block_size, embedding_size, Rank3Matrix<TypeParam>{});
        auto rotation_coeffts_block_mem = get_block_memory(block_size, embedding_size, Rank2Matrix<float>{});

        std::mt19937 engine;
        engine.seed(0);
        std::uniform_real_distribution<float> rng(-2.0, 2.0);

        auto generate_fn = [&]() {
            return TypeParam(rng(engine));
        };

        std::generate(cache_block_mem_ref.begin(), cache_block_mem_ref.end(), generate_fn);
        // coeffts are now not strictly sine-cosine pairs, but it does not matter for the kernels
        std::generate(rotation_coeffts_block_mem.begin(),
                      rotation_coeffts_block_mem.end(),
                      generate_fn);



        auto cache_block_mem_hw = cache_block_mem_ref;

        auto raw_mem_ptr_ref = cache_block_mem_ref.data();
        auto raw_rotation_coefficients_mem_ptr = rotation_coeffts_block_mem.data();
        auto raw_mem_ptr_hw = cache_block_mem_hw.data();

        ov::intel_cpu::PerfCount counter;
        {
            ov::intel_cpu::PerfHelper helper(counter);
            rotate_kv_cache_block_opt(raw_mem_ptr_hw,
                                     raw_rotation_coefficients_mem_ptr,
                                     num_heads,
                                     block_size,
                                     embedding_size);
        }

        {
            ov::intel_cpu::PerfHelper helper(counter);
            rotate_kv_cache_block_ref(raw_mem_ptr_ref,
                                     raw_rotation_coefficients_mem_ptr,
                                     num_heads,
                                     block_size,
                                     embedding_size);
        }

        auto ref_values_after_rotation = get_matrix_from_mem(cache_block_mem_ref, num_heads, block_size, embedding_size);
        auto opt_values_after_rotation = get_matrix_from_mem(cache_block_mem_hw, num_heads, block_size, embedding_size);
        compare_with_tolerance(opt_values_after_rotation, ref_values_after_rotation, get_tolerance<TypeParam>());
    }
};

using OV_FP_TYPES = ::testing::Types<float, ov::float16, ov::bfloat16>;

TYPED_TEST_SUITE_P(CacheRotationKernelInputTypeParameterizedTest);

TYPED_TEST_P(CacheRotationKernelInputTypeParameterizedTest, RefBlockRotationGivesReferenceResults) {
    auto raw_cache_mem_ptr = this->cache_mem.data();
    auto raw_rotation_coefficients_mem_ptr = this->rotation_coefficients_mem.data();

    rotate_kv_cache_block_ref(raw_cache_mem_ptr,
                             raw_rotation_coefficients_mem_ptr,
                             this->num_heads,
                             this->block_size,
                             this->embedding_size);

    auto test_values_after_rotation =
        get_matrix_from_mem(this->cache_mem, this->num_heads, this->block_size, this->embedding_size);
    compare_with_tolerance(test_values_after_rotation, this->ref_values_after_rotation, get_tolerance<TypeParam>());
}

enum class TargetInstructionSet { AVX2, AVX512 };

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override" // false positive in gtest macro internals
#endif

MATCHER_P3(IsNFirstValuesNear, ref_container, abs_err, n, "") {
    if (ref_container.size() < n || arg.size() < n)
        return false;
    if (ref_container.size() != arg.size())
        return false;

    bool is_ok = true;
    for (size_t i = 0; i < n; i++) {
        if (!::testing::ExplainMatchResult(::testing::FloatNear(static_cast<float>(arg[i]), abs_err),
                                           static_cast<float>(ref_container[i]),
                                           result_listener)) {
            *result_listener << " for element at idx " << i << '\n';
            is_ok = false;
        }
    }
    return is_ok;
}


#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

class CacheRotationKernelInstructionParameterizedTest
    : public ::testing::TestWithParam<std::tuple<TargetInstructionSet, size_t>> {
protected:
    constexpr static size_t MAX_CHUNK_SIZE_IN_ELEMENTS = 16;
    template <class T>
    using MemChunk = std::array<T, MAX_CHUNK_SIZE_IN_ELEMENTS>;

    template <class T>
    void test_chunk_rotation_for_type() {
        auto instruction_set = std::get<0>(GetParam());
        if (instruction_set == TargetInstructionSet::AVX512 && (!ov::with_cpu_x86_avx512f())) {
            GTEST_SKIP() << "test executor must have AVX512 support";
        }
        if (instruction_set == TargetInstructionSet::AVX2 && (!ov::with_cpu_x86_avx2())) {
            GTEST_SKIP() << "test executor must have AVX2 support";
        }
        auto num_elements_to_process = std::get<1>(GetParam());

        MemChunk<T> chunk_x = {-0.76777814f,
                               0.97583583f,
                               -0.23619731f,
                               0.19022397f,
                               0.56691264f,
                               0.64870757f,
                               0.63334306f,
                               1.97307894f,
                               0.72495168f,
                               1.22328697f,
                               -0.6005607f,
                               0.17189973f,
                               -0.92268487f,
                               0.40205632f,
                               0.85996431f,
                               1.70078315f};

        MemChunk<T> chunk_y = {1.68812157f,
                               -0.90722836f,
                               0.58474063f,
                               -0.64561766f,
                               0.62651501f,
                               1.55990472f,
                               0.41571189f,
                               0.38366555f,
                               0.09841767f,
                               0.02218336f,
                               -0.07657361f,
                               1.6062845f,
                               -1.08282323f,
                               -0.92034808f,
                               -1.48428038f,
                               0.43501142f};

        MemChunk<float> chunk_cos = {-0.87461971f,
                                     0.95630476f,
                                     0.08715574f,
                                     0.8480481f,
                                     -0.9612617f,
                                     0.27563736f,
                                     0.97437006f,
                                     0.66913061f,
                                     -0.89100652f,
                                     0.98480775f,
                                     -0.7313537f,
                                     -0.2419219f,
                                     0.10452846f,
                                     0.70710678f,
                                     -0.32556815f,
                                     -0.2923717f};

        MemChunk<float> chunk_sin = {-0.48480962f,
                                     -0.2923717f,
                                     0.9961947f,
                                     0.52991926f,
                                     0.27563736f,
                                     -0.9612617f,
                                     -0.22495105f,
                                     0.74314483f,
                                     0.4539905f,
                                     -0.17364818f,
                                     -0.68199836f,
                                     -0.97029573f,
                                     -0.9945219f,
                                     -0.70710678f,
                                     -0.94551858f,
                                     0.95630476f};

        MemChunk<float> ref_chunk_cos = chunk_cos;
        MemChunk<float> ref_chunk_sin = chunk_sin;

        MemChunk<T> ref_chunk_x = {1.48993147f,
                                   0.66794854f,
                                   -0.60310147f,
                                   0.50344431f,
                                   -0.71764235f,
                                   1.6782847f,
                                   0.71062535f,
                                   1.03512844f,
                                   -0.69061736f,
                                   1.20855459f,
                                   0.38699921f,
                                   1.51698468f,
                                   -1.17333824f,
                                   -0.36648762f,
                                   -1.68339166f,
                                   -0.91326436f};

        MemChunk<T> ref_chunk_y = {-1.10423816f,
                                   -1.15289358f,
                                   -0.184335f,
                                   -0.44671148f,
                                   -0.44598258f,
                                   -0.19360973f,
                                   0.26258603f,
                                   1.72300577f,
                                   0.24143039f,
                                   -0.19057521f,
                                   0.46558381f,
                                   -0.55538896f,
                                   0.80444446f,
                                   -0.93508112f,
                                   -0.32987781f,
                                   1.49928198f};

        // unprocessed elements should remain untouched
        std::copy(chunk_x.begin() + num_elements_to_process,
                  chunk_x.end(),
                  ref_chunk_x.begin() + num_elements_to_process);
        std::copy(chunk_y.begin() + num_elements_to_process,
                  chunk_y.end(),
                  ref_chunk_y.begin() + num_elements_to_process);

        switch (instruction_set) {
            using namespace ov::Extensions::Cpu::XARCH;
#if defined(HAVE_AVX2)
        case TargetInstructionSet::AVX2:
            rotate_kv_cache_chunk_avx2(chunk_x.data(),
                                       chunk_y.data(),
                                       chunk_cos.data(),
                                       chunk_sin.data(),
                                       num_elements_to_process,
                                       /* is_tail = */ num_elements_to_process < vec_len_f32_avx2);
            break;
#endif
#if defined(HAVE_AVX512F)
        case TargetInstructionSet::AVX512:
            rotate_kv_cache_chunk_avx512(chunk_x.data(),
                                         chunk_y.data(),
                                         chunk_cos.data(),
                                         chunk_sin.data(),
                                         num_elements_to_process,
                                         /* is_tail = */ num_elements_to_process < vec_len_f32_avx512);
            break;
#endif
        default:
            FAIL() << "unknown target instruction set";
        }

        std::string type_name = ov::element::from<T>().to_string();

        EXPECT_THAT(chunk_x, IsNFirstValuesNear(ref_chunk_x, get_tolerance<T>(), num_elements_to_process))
            << ", element type is: " << type_name;
        EXPECT_THAT(chunk_y, IsNFirstValuesNear(ref_chunk_y, get_tolerance<T>(), num_elements_to_process))
            << ", element type is: " << type_name;

        EXPECT_EQ(chunk_cos, ref_chunk_cos) << ", element type is: " << type_name;
        EXPECT_EQ(chunk_sin, ref_chunk_sin) << ", element type is: " << type_name;
    }
};

TEST_P(CacheRotationKernelInstructionParameterizedTest, OptChunkRotationGivesReferenceResults) {
    test_chunk_rotation_for_type<float>();
    test_chunk_rotation_for_type<ov::float16>();
    test_chunk_rotation_for_type<ov::bfloat16>();
}

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
auto TEST_STRUCT_TO_NAME_FN =
    [](const testing::TestParamInfo<CacheRotationKernelInstructionParameterizedTest::ParamType>& info) {
        size_t num_elts = std::get<1>(info.param);
        switch (std::get<0>(info.param)) {
        case TargetInstructionSet::AVX2:
            return std::string("avx2-") + std::to_string(num_elts);
        case TargetInstructionSet::AVX512:
            return std::string("avx512-") + std::to_string(num_elts);
        }
        return std::string("unknown");
    };
#endif

#if defined(HAVE_AVX2)
INSTANTIATE_TEST_SUITE_P(AVX2,
                         CacheRotationKernelInstructionParameterizedTest,
                         ::testing::Combine(::testing::Values(TargetInstructionSet::AVX2),
                                            ::testing::Range(size_t(0),
                                                             ov::Extensions::Cpu::XARCH::vec_len_f32_avx2 + 1)),
                         TEST_STRUCT_TO_NAME_FN);
#endif
#if defined(HAVE_AVX512F)
INSTANTIATE_TEST_SUITE_P(AVX512,
                         CacheRotationKernelInstructionParameterizedTest,
                         ::testing::Combine(::testing::Values(TargetInstructionSet::AVX512),
                                            ::testing::Range(size_t(0),
                                                             ov::Extensions::Cpu::XARCH::vec_len_f32_avx512 + 1)),
                         TEST_STRUCT_TO_NAME_FN);
#endif

TYPED_TEST_P(CacheRotationKernelInputTypeParameterizedTest, OptBlockRotationGivesReferenceResults) {
    auto raw_cache_mem_ptr = this->cache_mem.data();
    auto raw_rotation_coefficients_mem_ptr = this->rotation_coefficients_mem.data();

    rotate_kv_cache_block_opt(raw_cache_mem_ptr,
                             raw_rotation_coefficients_mem_ptr,
                             this->num_heads,
                             this->block_size,
                             this->embedding_size);

    auto test_values_after_rotation =
        get_matrix_from_mem(this->cache_mem, this->num_heads, this->block_size, this->embedding_size);
    compare_with_tolerance(test_values_after_rotation, this->ref_values_after_rotation, get_tolerance<TypeParam>());
}

TYPED_TEST_P(CacheRotationKernelInputTypeParameterizedTest, OptBlockRotationIsSimilarToRef) {
    // short case
    this->test_block_opt_vs_ref(/* num_heads = */ 4, /* embedding_size = */ 64, /* block_size = */ 2);

    // long case
    this->test_block_opt_vs_ref(256, 1024, 32);
}

REGISTER_TYPED_TEST_SUITE_P(CacheRotationKernelInputTypeParameterizedTest,
                            RefBlockRotationGivesReferenceResults,
                            OptBlockRotationGivesReferenceResults,
                            OptBlockRotationIsSimilarToRef);
INSTANTIATE_TYPED_TEST_SUITE_P(AllFPTypes, CacheRotationKernelInputTypeParameterizedTest, OV_FP_TYPES);
