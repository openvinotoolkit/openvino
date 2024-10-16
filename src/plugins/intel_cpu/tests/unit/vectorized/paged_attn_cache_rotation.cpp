// Copyright (C) 2018-2024 Intel Corporation
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
std::shared_ptr<T[]> get_block_memory(size_t block_size, size_t embedding_size, const Rank2Matrix<T>& init_values) {
    auto mem = std::shared_ptr<T[]>(new T[block_size * embedding_size]);
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
std::shared_ptr<T[]> get_block_memory(size_t num_heads,
                                      size_t block_size,
                                      size_t embedding_size,
                                      const Rank3Matrix<T>& init_values) {
    auto mem = std::shared_ptr<T[]>(new T[num_heads * block_size * embedding_size]);
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
Rank3Matrix<T> get_matrix_from_mem(std::shared_ptr<T[]> mem_ptr,
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
                retval[i][j][k] = mem_ptr[block_size * embedding_size * i + embedding_size * j + k];
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
    return 1e-6;
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
class CacheRotationKernelTest : public ::testing::Test {
public:
    void SetUp() override {
        Rank3Matrix<TypeParam> values_before_rotation = {
            {
                {1.0, 1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0, 1.0},
            },
            {
                {-2.0, -2.0, -2.0, -2.0},
                {2.0, 2.0, 2.0, 2.0},
                {-1.0, 2.0, -3.0, 4.0},
                {2.0, 2.0, 2.0, 2.0},
            },
        };
        cache_mem_ptr = get_block_memory(num_heads, block_size, embedding_size, values_before_rotation);

        Rank2Matrix<float> rotation_values = {
            {0.5, 0.70710678, 0.86602540, -0.70710678},
            {0.86602540, 1.0, 0.5, 0.0},
            {-0.70710678, 0.0, 0.70710678, 1.0},
            {0.0, 0.6, -1.0, -0.8},
        };

        rotation_coefficients_mem_ptr = get_block_memory(block_size, embedding_size, rotation_values);
    }
    size_t num_heads = 2;
    size_t block_size = 4;
    size_t embedding_size = 4;
    std::shared_ptr<TypeParam[]> cache_mem_ptr;
    std::shared_ptr<float[]> rotation_coefficients_mem_ptr;
    Rank3Matrix<TypeParam> ref_values_after_rotation = {
        {
            {-0.36602540, 1.41421356, 1.36602540, 0.00000000},
            {0.36602540, 1.00000000, 1.36602540, 1.00000000},
            {-1.41421356, -1.00000000, 0.00000000, 1.00000000},
            {1.00000000, 1.40000000, -1.00000000, -0.20000000},
        },
        {
            {0.73205081, -2.82842712, -2.73205081, 0.00000000},
            {0.73205081, 2.00000000, 2.73205081, 2.00000000},
            {2.82842712, -4.00000000, 1.41421356, 2.00000000},
            {2.00000000, 2.80000000, -2.00000000, -0.40000000},
        },
    };

    void test_block_hw_vs_sw(size_t num_heads, size_t embedding_size, size_t block_size) {
        auto cache_block_mem_sw = get_block_memory(num_heads, block_size, embedding_size, Rank3Matrix<TypeParam>{});
        auto rotation_coeffts_block_mem = get_block_memory(block_size, embedding_size, Rank2Matrix<float>{});

        std::mt19937 engine;
        engine.seed(0);
        std::uniform_real_distribution<float> rng(-2.0, 2.0);

        auto raw_mem_ptr_sw = cache_block_mem_sw.get();
        auto raw_rotation_coefficients_mem_ptr = rotation_coeffts_block_mem.get();

        auto generate_fn = [&]() {
            return TypeParam(rng(engine));
        };

        std::generate(raw_mem_ptr_sw, raw_mem_ptr_sw + num_heads * block_size * embedding_size, generate_fn);
        // coeffts are now not strictly sine-cosine pairs, but it does not matter for the kernels
        std::generate(raw_rotation_coefficients_mem_ptr,
                      raw_rotation_coefficients_mem_ptr + block_size * embedding_size,
                      generate_fn);

        auto cache_block_mem_hw = get_block_memory(num_heads, block_size, embedding_size, Rank3Matrix<TypeParam>{});
        auto raw_mem_ptr_hw = cache_block_mem_hw.get();
        std::copy(raw_mem_ptr_sw, raw_mem_ptr_sw + num_heads * block_size * embedding_size, raw_mem_ptr_hw);

        ov::intel_cpu::PerfCount counter;
        {
            ov::intel_cpu::PerfHelper helper(counter);
            rotate_kv_cache_block_hw(raw_mem_ptr_hw,
                                     rotation_coeffts_block_mem.get(),
                                     num_heads,
                                     block_size,
                                     embedding_size);
        }

        {
            ov::intel_cpu::PerfHelper helper(counter);
            rotate_kv_cache_block_sw(raw_mem_ptr_sw,
                                     rotation_coeffts_block_mem.get(),
                                     num_heads,
                                     block_size,
                                     embedding_size);
        }

        auto sw_values_after_rotation = get_matrix_from_mem(cache_block_mem_sw, num_heads, block_size, embedding_size);
        auto hw_values_after_rotation = get_matrix_from_mem(cache_block_mem_hw, num_heads, block_size, embedding_size);
        compare_with_tolerance(hw_values_after_rotation, sw_values_after_rotation, get_tolerance<TypeParam>());
    }
};

using OV_FP_TYPES = ::testing::Types<float, ov::float16, ov::bfloat16>;

TYPED_TEST_SUITE(CacheRotationKernelTest, OV_FP_TYPES);

TYPED_TEST(CacheRotationKernelTest, SWBlockRotationGivesReferenceResults) {
    auto raw_cache_mem_ptr = this->cache_mem_ptr.get();
    auto raw_rotation_coefficients_mem_ptr = this->rotation_coefficients_mem_ptr.get();

    rotate_kv_cache_block_sw(raw_cache_mem_ptr,
                             raw_rotation_coefficients_mem_ptr,
                             this->num_heads,
                             this->block_size,
                             this->embedding_size);

    auto test_values_after_rotation =
        get_matrix_from_mem(this->cache_mem_ptr, this->num_heads, this->block_size, this->embedding_size);
    compare_with_tolerance(test_values_after_rotation, this->ref_values_after_rotation, get_tolerance<TypeParam>());
}

enum class TargetInstructionSet { AVX2, AVX512 };

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

class CacheRotationHWKernelTest : public ::testing::TestWithParam<std::tuple<TargetInstructionSet, size_t>> {
protected:
    constexpr static size_t MAX_CHUNK_SIZE_IN_ELEMENTS = 16;
    template <class T>
    using MemChunk = std::array<T, MAX_CHUNK_SIZE_IN_ELEMENTS>;

    template <class T>
    void test_chunk_rotation_for_type() {
        auto instruction_set = std::get<0>(GetParam());
        auto num_elements_to_process = std::get<1>(GetParam());

        MemChunk<T> chunk_x = {-0.76777814,
                               0.97583583,
                               -0.23619731,
                               0.19022397,
                               0.56691264,
                               0.64870757,
                               0.63334306,
                               1.97307894,
                               0.72495168,
                               1.22328697,
                               -0.6005607,
                               0.17189973,
                               -0.92268487,
                               0.40205632,
                               0.85996431,
                               1.70078315};

        MemChunk<T> chunk_y = {1.68812157,
                               -0.90722836,
                               0.58474063,
                               -0.64561766,
                               0.62651501,
                               1.55990472,
                               0.41571189,
                               0.38366555,
                               0.09841767,
                               0.02218336,
                               -0.07657361,
                               1.6062845,
                               -1.08282323,
                               -0.92034808,
                               -1.48428038,
                               0.43501142};

        MemChunk<float> chunk_cos = {-0.87461971,
                                     0.95630476,
                                     0.08715574,
                                     0.8480481,
                                     -0.9612617,
                                     0.27563736,
                                     0.97437006,
                                     0.66913061,
                                     -0.89100652,
                                     0.98480775,
                                     -0.7313537,
                                     -0.2419219,
                                     0.10452846,
                                     0.70710678,
                                     -0.32556815,
                                     -0.2923717};

        MemChunk<float> chunk_sin = {-0.48480962,
                                     -0.2923717,
                                     0.9961947,
                                     0.52991926,
                                     0.27563736,
                                     -0.9612617,
                                     -0.22495105,
                                     0.74314483,
                                     0.4539905,
                                     -0.17364818,
                                     -0.68199836,
                                     -0.97029573,
                                     -0.9945219,
                                     -0.70710678,
                                     -0.94551858,
                                     0.95630476};

        MemChunk<float> ref_chunk_cos = chunk_cos;
        MemChunk<float> ref_chunk_sin = chunk_sin;

        MemChunk<T> ref_chunk_x = {1.48993147,
                                   0.66794854,
                                   -0.60310147,
                                   0.50344431,
                                   -0.71764235,
                                   1.6782847,
                                   0.71062535,
                                   1.03512844,
                                   -0.69061736,
                                   1.20855459,
                                   0.38699921,
                                   1.51698468,
                                   -1.17333824,
                                   -0.36648762,
                                   -1.68339166,
                                   -0.91326436};

        MemChunk<T> ref_chunk_y = {-1.10423816,
                                   -1.15289358,
                                   -0.184335,
                                   -0.44671148,
                                   -0.44598258,
                                   -0.19360973,
                                   0.26258603,
                                   1.72300577,
                                   0.24143039,
                                   -0.19057521,
                                   0.46558381,
                                   -0.55538896,
                                   0.80444446,
                                   -0.93508112,
                                   -0.32987781,
                                   1.49928198};

        // unprocessed elements should remain untouched
        std::copy(chunk_x.begin() + num_elements_to_process,
                  chunk_x.end(),
                  ref_chunk_x.begin() + num_elements_to_process);
        std::copy(chunk_y.begin() + num_elements_to_process,
                  chunk_y.end(),
                  ref_chunk_y.begin() + num_elements_to_process);

        switch (instruction_set) {
            using namespace ov::Extensions::Cpu::XARCH;
        case TargetInstructionSet::AVX2:
            rotate_kv_cache_chunk_avx2(chunk_x.data(),
                                       chunk_y.data(),
                                       chunk_cos.data(),
                                       chunk_sin.data(),
                                       num_elements_to_process,
                                       /* is_underutilizing = */ num_elements_to_process < vec_len_f32_avx2);
            break;
        case TargetInstructionSet::AVX512:
            rotate_kv_cache_chunk_avx512(chunk_x.data(),
                                         chunk_y.data(),
                                         chunk_cos.data(),
                                         chunk_sin.data(),
                                         num_elements_to_process,
                                         /* is_underutilizing = */ num_elements_to_process < vec_len_f32_avx512);
            break;
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

TEST_P(CacheRotationHWKernelTest, HWChunkRotationGivesReferenceResults) {
    test_chunk_rotation_for_type<float>();
    test_chunk_rotation_for_type<ov::float16>();
    test_chunk_rotation_for_type<ov::bfloat16>();
}

auto TEST_STRUCT_TO_NAME_FN = [](const testing::TestParamInfo<CacheRotationHWKernelTest::ParamType>& info) {
    size_t num_elts = std::get<1>(info.param);
    switch (std::get<0>(info.param)) {
    case TargetInstructionSet::AVX2:
        return std::string("avx2-") + std::to_string(num_elts);
    case TargetInstructionSet::AVX512:
        return std::string("avx512-") + std::to_string(num_elts);
    }
    return std::string("unknown");
};

INSTANTIATE_TEST_SUITE_P(AVX2,
                         CacheRotationHWKernelTest,
                         ::testing::Combine(::testing::Values(TargetInstructionSet::AVX2),
                                            ::testing::Range(size_t(0),
                                                             ov::Extensions::Cpu::XARCH::vec_len_f32_avx2 + 1)),
                         TEST_STRUCT_TO_NAME_FN);
INSTANTIATE_TEST_SUITE_P(AVX512,
                         CacheRotationHWKernelTest,
                         ::testing::Combine(::testing::Values(TargetInstructionSet::AVX512),
                                            ::testing::Range(size_t(0),
                                                             ov::Extensions::Cpu::XARCH::vec_len_f32_avx512 + 1)),
                         TEST_STRUCT_TO_NAME_FN);

TYPED_TEST(CacheRotationKernelTest, HWBlockRotationGivesReferenceResults) {
    auto raw_cache_mem_ptr = this->cache_mem_ptr.get();
    auto raw_rotation_coefficients_mem_ptr = this->rotation_coefficients_mem_ptr.get();

    rotate_kv_cache_block_hw(raw_cache_mem_ptr,
                             raw_rotation_coefficients_mem_ptr,
                             this->num_heads,
                             this->block_size,
                             this->embedding_size);

    auto test_values_after_rotation =
        get_matrix_from_mem(this->cache_mem_ptr, this->num_heads, this->block_size, this->embedding_size);
    compare_with_tolerance(test_values_after_rotation, this->ref_values_after_rotation, get_tolerance<TypeParam>());
}

TYPED_TEST(CacheRotationKernelTest, HWBlockRotationIsSimilarToSW) {
    // short case
    this->test_block_hw_vs_sw(/* num_heads = */ 4, /* embedding_size = */ 64, /* block_size = */ 2);

    // long case
    this->test_block_hw_vs_sw(256, 1024, 32);
}
