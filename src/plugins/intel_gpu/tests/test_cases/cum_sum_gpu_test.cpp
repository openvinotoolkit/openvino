// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/cum_sum.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <algorithm>
#include <fstream>

using namespace cldnn;
using namespace ::tests;

template<typename T = float>
static std::vector<T> cumsum(const std::vector<T>& input,
                             const cldnn::format& format,
                             const std::vector<int>& shape,
                             const int axis = 0,
                             bool exclusive = false,
                             bool reverse = false) {
    std::vector<T> output(input.size());
    int dimNum = 0;
    std::vector<int> reordered_shape = shape;

    if (format == format::bfwzyx) {
        dimNum = 6;
    } else if (format == format::bfzyx) {
        dimNum = 5;
        for (int i = 2; i < dimNum; ++i) {
            reordered_shape[i] = shape[i + 1];
        }
    } else {
        dimNum = 4;
        for (int i = 2; i < dimNum; ++i) {
            reordered_shape[i] = shape[i + 2];
        }
    }

    std::vector<int> sizeDim(dimNum);
    sizeDim[dimNum - 1] = 1;
    for (size_t i = dimNum - 1, mult = 1; i > 0; --i) {
        mult *= reordered_shape[i];
        sizeDim[i - 1] = static_cast<int>(mult);
    }

    auto getFullIndex = [&sizeDim](int ind) {
        std::vector<int> fullInd(sizeDim.size());
        fullInd[0] = ind / sizeDim[0];
        for (int i = 1, numItems = 0; i < static_cast<int>(fullInd.size()); ++i) {
            numItems += fullInd[i - 1] * sizeDim[i - 1];
            fullInd[i] = (ind - numItems)/sizeDim[i];
        }
        return fullInd;
    };

    auto getIndex = [&sizeDim](std::vector<int> fullInd) {
        size_t index = 0;
        for (size_t i = 0; i < fullInd.size(); ++i) {
            index += fullInd[i] * sizeDim[i];
        }
        return index;
    };

    for (int i = 0; i < static_cast<int>(output.size()); ++i) {
        auto fullInd = getFullIndex(i);

        int stopInd = fullInd[axis] + 1;
        if (reverse) {
            stopInd = reordered_shape[axis];
            if (exclusive) {
                ++fullInd[axis];
            }
        } else {
            fullInd[axis] = 0;
            if (exclusive) {
                --stopInd;
            }
        }

        T res = (T)0;
        for (; fullInd[axis] < stopInd; ++fullInd[axis]) {
            auto ind = getIndex(fullInd);
            res += input[ind];
        }

        output[i] = (T)res;
    }
    return output;
}

template<typename T1, typename T2>
static std::vector<T1> vectorCast(const std::vector<T2>& vec) {
    std::vector<T1> ret(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        ret[i] = T1(vec[i]);
    }
    return ret;
}

static cldnn::cum_sum::cum_sum_axis getCumSumAxis(int axis, unsigned sz) {
    unsigned cldnn_axis = axis;
    if (axis >= 2) {
        auto spatial_axis = axis - 2;
        auto spatial_size = std::max(sz, 4u) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }
    switch (cldnn_axis) {
        case 0:
            return cldnn::cum_sum::cum_sum_axis::along_b;
        case 1:
            return cldnn::cum_sum::cum_sum_axis::along_f;
        case 2:
            return cldnn::cum_sum::cum_sum_axis::along_x;
        case 3:
            return cldnn::cum_sum::cum_sum_axis::along_y;
        case 4:
            return cldnn::cum_sum::cum_sum_axis::along_z;
        case 5:
            return cldnn::cum_sum::cum_sum_axis::along_w;
        default:
            return cldnn::cum_sum::cum_sum_axis::along_b;
    }
}

#define CASE_CUM_SUM_AXIS_0 ::testing::Values(5), ::testing::Values(1), ::testing::Values(1), \
                            ::testing::Values(1), ::testing::Values(1), ::testing::Values(1), \
                            ::testing::Values(format::bfyx), ::testing::ValuesIn(axes[0]),    \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants)
#define CASE_CUM_SUM_AXIS_1 ::testing::Values(2), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(1), ::testing::Values(1), ::testing::Values(1), \
                            ::testing::Values(format::bfyx), ::testing::ValuesIn(axes[1]),    \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants)
#define CASE_CUM_SUM_AXIS_2 ::testing::Values(5), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(1), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(format::bfyx), ::testing::ValuesIn(axes[2]),    \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants)
#define CASE_CUM_SUM_AXIS_3 ::testing::Values(5), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(1), ::testing::Values(5), ::testing::Values(5), \
                            ::testing::Values(format::bfyx), ::testing::ValuesIn(axes[3]),    \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants)
#define CASE_CUM_SUM_AXIS_4 ::testing::Values(5), ::testing::Values(5), ::testing::Values(1), \
                            ::testing::Values(5), ::testing::Values(5), ::testing::Values(5), \
                            ::testing::Values(format::bfzyx), ::testing::ValuesIn(axes[4]),   \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants)
#define CASE_CUM_SUM_AXIS_5 ::testing::Values(5), ::testing::Values(5), ::testing::Values(5), \
                            ::testing::Values(5), ::testing::Values(5), ::testing::Values(5), \
                            ::testing::Values(format::bfwzyx), ::testing::ValuesIn(axes[5]),  \
                            ::testing::ValuesIn(variants), ::testing::ValuesIn(variants)

using cum_sum_test_params = std::tuple<int,            // batch
                                       int,            // feature
                                       int,            // w
                                       int,            // z
                                       int,            // y
                                       int,            // x
                                       cldnn::format,  // in_out format
                                       int,            // axis
                                       bool,           // exclusive
                                       bool>;          // reverse

template <typename cum_sum_params, typename input_type = float, typename output_type = float>
class cum_sum_gpu : public ::testing::TestWithParam<cum_sum_params> {
public:

    data_types get_alloc_data_type(void) {
        if (std::is_same<input_type, float>::value)
            return data_types::f32;
        else if (std::is_same<input_type, FLOAT16>::value)
            return data_types::f16;
        else if (std::is_same<input_type, int32_t>::value)
            return data_types::i32;
        else if (std::is_same<input_type, int64_t>::value)
            return data_types::i64;
        else
            throw std::runtime_error("Unsupported cum sum data type in cum_sum_gpu_test.cpp");
    }

    void execute(cum_sum_params& p) {
        auto& engine = get_test_engine();

        auto b = std::get<0>(p);
        auto f = std::get<1>(p);
        auto w = std::get<2>(p);
        auto z = std::get<3>(p);
        auto y = std::get<4>(p);
        auto x = std::get<5>(p);
        tensor shape = tensor{ batch(b), feature(f), spatial(x, y, z, w) };

        auto in_out_format = std::get<6>(p);
        auto axis = std::get<7>(p);
        auto exclusive = std::get<8>(p);
        auto reverse = std::get<9>(p);

        auto size = 4;
        if (in_out_format == format::bfzyx)
            size = 5;
        else if (in_out_format == format::bfwzyx)
            size = 6;

        auto input = engine.allocate_memory({ get_alloc_data_type(), in_out_format, shape });
        const int inputSize = b * f * w * z * y * x;
        VF<input_type> inputVals = std::is_same<input_type, FLOAT16>::value ?
                                   generate_random_1d<input_type>(inputSize, -1, 1, 1) :
                                   generate_random_1d<input_type>(inputSize, -100, 100, 8);

        set_values(input, inputVals);

        topology topology;
        topology.add(input_layout("Input0", input->get_layout()));
        topology.add(cum_sum("cum_sum", "Input0", getCumSumAxis(axis, size), exclusive, reverse));

        network network(engine, topology);

        network.set_input_data("Input0", input);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "cum_sum");

        auto output = outputs.at("cum_sum").get_memory();
        cldnn::mem_lock<output_type> output_ptr(output, get_test_stream());

        auto answers = cumsum<output_type>(inputVals, in_out_format, { b, f, w, z, y, x }, axis, exclusive, reverse);
        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i) {
            EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
        }
    }
};

class cum_sum_gpu_fp16 : public ::cum_sum_gpu<cum_sum_test_params, FLOAT16, FLOAT16> {};
class cum_sum_gpu_fp32 : public ::cum_sum_gpu<cum_sum_test_params, float, float> {};
class cum_sum_gpu_int32 : public ::cum_sum_gpu<cum_sum_test_params, int32_t, int32_t> {};
class cum_sum_gpu_int64 : public ::cum_sum_gpu<cum_sum_test_params, int64_t, int64_t> {};

TEST_P(cum_sum_gpu_fp16, basic) { auto p = GetParam(); execute(p); }
TEST_P(cum_sum_gpu_fp32, basic) { auto p = GetParam(); execute(p); }
TEST_P(cum_sum_gpu_int32, basic) { auto p = GetParam(); execute(p); }
TEST_P(cum_sum_gpu_int64, basic) { auto p = GetParam(); execute(p); }

namespace {
    std::vector<std::vector<int>> axes = {
        { 0 },
        { 0, 1 },
        { 0, 1, 2 },
        { 0, 1, 2, 3 },
        { 0, 1, 2, 3, 4 },
        { 0, 1, 2, 3, 4, 5 },
    };
    std::vector<bool> variants = { false, true };
}

INSTANTIATE_TEST_SUITE_P(axis_0, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_0));
INSTANTIATE_TEST_SUITE_P(axis_0, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_0));
INSTANTIATE_TEST_SUITE_P(axis_0, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_0));
INSTANTIATE_TEST_SUITE_P(axis_0, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_0));

INSTANTIATE_TEST_SUITE_P(axis_1, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_1));
INSTANTIATE_TEST_SUITE_P(axis_1, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_1));
INSTANTIATE_TEST_SUITE_P(axis_1, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_1));
INSTANTIATE_TEST_SUITE_P(axis_1, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_1));

INSTANTIATE_TEST_SUITE_P(axis_2, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_2));
INSTANTIATE_TEST_SUITE_P(axis_2, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_2));
INSTANTIATE_TEST_SUITE_P(axis_2, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_2));
INSTANTIATE_TEST_SUITE_P(axis_2, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_2));

INSTANTIATE_TEST_SUITE_P(axis_3, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_3));
INSTANTIATE_TEST_SUITE_P(axis_3, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_3));
INSTANTIATE_TEST_SUITE_P(axis_3, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_3));
INSTANTIATE_TEST_SUITE_P(axis_3, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_3));

INSTANTIATE_TEST_SUITE_P(axis_4, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_4));
INSTANTIATE_TEST_SUITE_P(axis_4, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_4));
INSTANTIATE_TEST_SUITE_P(axis_4, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_4));
INSTANTIATE_TEST_SUITE_P(axis_4, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_4));

INSTANTIATE_TEST_SUITE_P(axis_5, cum_sum_gpu_fp16, ::testing::Combine(CASE_CUM_SUM_AXIS_5));
INSTANTIATE_TEST_SUITE_P(axis_5, cum_sum_gpu_fp32, ::testing::Combine(CASE_CUM_SUM_AXIS_5));
INSTANTIATE_TEST_SUITE_P(axis_5, cum_sum_gpu_int32, ::testing::Combine(CASE_CUM_SUM_AXIS_5));
INSTANTIATE_TEST_SUITE_P(axis_5, cum_sum_gpu_int64, ::testing::Combine(CASE_CUM_SUM_AXIS_5));

// FIXME: This test fails on some driver versions. Looks like UB in impl or driver issue
TEST(cum_sum_gpu_f16, DISABLED_basic_1d) {
    // Input : 5x1x1x1
    // Output : 5x1x1x1

    auto& engine = get_test_engine();
    tensor shape = { 5, 1, 1, 1 };
    std::vector<float> inputVals = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f
    };
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, shape });

    set_values(input, vectorCast<FLOAT16>(inputVals));

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(cum_sum("cum_sum", "Input0"));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "cum_sum");

    auto output = outputs.at("cum_sum").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    auto answers = cumsum(inputVals, format::bfyx, { 5, 1, 1, 1, 1, 1 });

    ASSERT_EQ(output->count(), answers.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], float16_to_float32(output_ptr[i]))) << i;
    }
}
