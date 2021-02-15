/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/input_layout.hpp>
#include "api/cum_sum.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/data.hpp>

#include <algorithm>
#include <fstream>

using namespace cldnn;
using namespace tests;

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
            if (exclusive)
                ++fullInd[axis];
        }
        else {
            fullInd[axis] = 0;
            if (exclusive) {
                --stopInd;
            }
        }

        float res = 0.f;
        for (; fullInd[axis] < stopInd; ++fullInd[axis]) {
            auto ind = getIndex(fullInd);
            res += input[ind];
        }
        output[i] = res;
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

template<typename T = float>
static std::vector<T> generateVector(size_t sz) {
    std::vector<T> vec(sz);
    T n = 0;
    std::generate(vec.begin(), vec.end(), [&n]() {
            return n++;
        });
    return vec;
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
class cum_sum_gpu : public ::testing::TestWithParam<cum_sum_test_params> {};

TEST_P(cum_sum_gpu, basic_test) {
    auto p = GetParam();
    const auto& engine = get_test_engine();

    auto b = std::get<0>(p);
    auto f = std::get<1>(p);
    auto w = std::get<2>(p);
    auto z = std::get<3>(p);
    auto y = std::get<4>(p);
    auto x = std::get<5>(p);
    tensor shape = tensor{batch(b), feature(f), spatial(x, y, z, w)};
    auto in_out_format = std::get<6>(p);
    auto axis = std::get<7>(p);
    auto exclusive = std::get<8>(p);
    auto reverse = std::get<9>(p);
    auto size = 4;
    if (in_out_format == format::bfwzyx)
        size = 6;
    else if (in_out_format == format::bfzyx)
        size = 5;

    auto input = memory::allocate(engine, { data_types::f32, in_out_format, shape });
    const int inputSize = b * f * w * z * y * x;
    auto inputVals = generateVector(inputSize);

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(cum_sum("cum_sum", "Input0", getCumSumAxis(axis, size), exclusive, reverse));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "cum_sum");

    auto output = outputs.at("cum_sum").get_memory();
    auto output_ptr = output.pointer<float>();

    auto answers = cumsum(inputVals, in_out_format, { b, f, w, z, y, x }, axis, exclusive, reverse);
    ASSERT_EQ(output_ptr.size(), answers.size());
    for (size_t i = 0; i < answers.size(); ++i)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

namespace {
    std::vector<std::vector<int>> axes = {
        {0},
        {0, 1},
        {0, 1, 2},
        {0, 1, 2, 3},
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4, 5},
    };
    std::vector<bool> variants = {false, true};
}
INSTANTIATE_TEST_CASE_P(
        axis_0,
        cum_sum_gpu,
        ::testing::Combine(
            ::testing::Values(5),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(format::bfyx),
            ::testing::ValuesIn(axes[0]),
            ::testing::ValuesIn(variants),
            ::testing::ValuesIn(variants)
            ), );

INSTANTIATE_TEST_CASE_P(
        axis_1,
        cum_sum_gpu,
        ::testing::Combine(
            ::testing::Values(2),
            ::testing::Values(5),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(format::bfyx),
            ::testing::ValuesIn(axes[1]),
            ::testing::ValuesIn(variants),
            ::testing::ValuesIn(variants)
            ), );

INSTANTIATE_TEST_CASE_P(
        axis_2,
        cum_sum_gpu,
        ::testing::Combine(
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(5),
            ::testing::Values(1),
            ::testing::Values(format::bfyx),
            ::testing::ValuesIn(axes[2]),
            ::testing::ValuesIn(variants),
            ::testing::ValuesIn(variants)
            ), );

INSTANTIATE_TEST_CASE_P(
        axis_3,
        cum_sum_gpu,
        ::testing::Combine(
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(1),
            ::testing::Values(1),
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(format::bfyx),
            ::testing::ValuesIn(axes[3]),
            ::testing::ValuesIn(variants),
            ::testing::ValuesIn(variants)
            ), );

INSTANTIATE_TEST_CASE_P(
        axis_4,
        cum_sum_gpu,
        ::testing::Combine(
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(1),
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(format::bfzyx),
            ::testing::ValuesIn(axes[4]),
            ::testing::ValuesIn(variants),
            ::testing::ValuesIn(variants)
            ), );

INSTANTIATE_TEST_CASE_P(
        axis_5,
        cum_sum_gpu,
        ::testing::Combine(
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(5),
            ::testing::Values(format::bfwzyx),
            ::testing::ValuesIn(axes[5]),
            ::testing::ValuesIn(variants),
            ::testing::ValuesIn(variants)
            ), );

TEST(cum_sum_gpu_f16, basic_1d) {
    // Input : 5x1x1x1
    // Output : 5x1x1x1

    const auto& engine = get_test_engine();
    tensor shape = { 5, 1, 1, 1 };
    std::vector<float> inputVals = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f
    };
    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, shape });

    set_values(input, vectorCast<FLOAT16>(inputVals));

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(cum_sum("cum_sum", "Input0"));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "cum_sum");

    auto output = outputs.at("cum_sum").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    auto answers = cumsum(inputVals, format::bfyx, { 5, 1, 1, 1, 1, 1 });

    ASSERT_EQ(output_ptr.size(), answers.size());
    for (size_t i = 0; i < answers.size(); ++i)
    {
        EXPECT_TRUE(are_equal(answers[i], float16_to_float32(output_ptr[i]))) << i;
    }
}

TEST(cum_sum_gpu_f32, perf) {
    // Input : 384x160x160x1
    // Output : 384x160x160x1

    constexpr int batch = 384;
    constexpr int features = 160;
    constexpr int y = 160;
    constexpr int x = 1;
    engine_configuration configuration(true);
    engine engine(configuration);
    tensor shape = { batch, features, y, x };
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, shape });
    constexpr int inputSize = batch * features * y * x;
    auto inputVals = generateVector(inputSize);

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(cum_sum("cum_sum", "Input0"));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "cum_sum");

    auto output = outputs.at("cum_sum").get_memory();
    auto output_ptr = output.pointer<float>();

    auto profilingTime = [](const primitive_id& id, const event& ev) {
            cldnn::instrumentation::profiling_info cldnnInfo{id, ev.get_profiling_info()};
            long long time = 0;
            for (auto &interval : cldnnInfo.intervals) {
                using duration_t = std::chrono::duration<long long, std::chrono::microseconds::period>;
                time += std::chrono::duration_cast<duration_t>(interval.value->value()).count();
            }
            return time;
        };


    auto ep = network.get_executed_primitives();
    auto cumSumEP = ep.find("cum_sum");
    ASSERT_NE(cumSumEP, ep.end()) << "Cannot find 'cum_sum' id in executed primitives";

    auto time = profilingTime(cumSumEP->first, cumSumEP->second);
    std::cout << "Time, id: " << cumSumEP->first << ", time: " << time << std::endl;
}
