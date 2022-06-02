// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "intel_gpu/primitives/roll.hpp"
#include "test_utils.h"

namespace cldnn {
template <>
struct type_to_data_type<FLOAT16> {
    static const data_types value = data_types::f16;
};
}  // namespace cldnn

using namespace cldnn;
using namespace tests;

namespace {

template <typename vecElementType>
std::string vec2str(const std::vector<vecElementType>& vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<vecElementType>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return "()";
}

template <class T>
struct roll_test_params {
    std::vector<int32_t> input_shape;
    std::vector<T> input_values;
    std::vector<int32_t> shift;
    std::vector<T> expected_values;
};

template <class T>
struct roll_test : testing::TestWithParam<roll_test_params<T>> {
    void test() {
        auto p = testing::TestWithParam<roll_test_params<T>>::GetParam();
        auto& engine = get_test_engine();

        const auto input_format = format::get_default_format(p.input_shape.size());
        const layout data_layout(type_to_data_type<T>::value, input_format, tensor(input_format, p.input_shape));
        auto input = engine.allocate_memory(data_layout);
        set_values(input, p.input_values);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(roll("roll", "input", tensor(input_format, p.shift)));

        network network(engine, topology);
        network.set_input_data("input", input);
        const auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "roll");

        auto output = outputs.at("roll").get_memory();
        cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), p.expected_values.size());
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            EXPECT_NEAR(p.expected_values[i], output_ptr[i], 1e-5f);
        }
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<roll_test_params<T>>& info) {
        auto& p = info.param;
        std::ostringstream result;
        result << "InputShape=" << vec2str(p.input_shape) << "_";
        result << "Precision=" << data_type_traits::name(type_to_data_type<T>::value) << "_";
        result << "Shift=" << vec2str(p.shift);
        return result.str();
    }
};

template <class T>
std::vector<roll_test_params<T>> getRollParams() {
    return {
        // from reference tests
        {{4, 3, 1, 1},                                                                              // Input shape
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},                                                   // Input values
         {2, 2, 0, 0},                                                                              // Shift
         {8, 9, 7, 11, 12, 10, 2, 3, 1, 5, 6, 4}},                                                  // Expected values
        {{4, 2, 3, 1},                                                                              // Input shape
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},   // Input values
         {0, 1, 1, 0},                                                                              // Shift
         {6, 4, 5, 3, 1, 2, 12, 10, 11, 9, 7, 8, 18, 16, 17, 15, 13, 14, 24, 22, 23, 21, 19, 20}},  // Expected values
        // from docs example
        {{4, 3, 1, 1},                              // Input shape
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},   // Input values
         {1, 0, 0, 0},                              // Shift
         {10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9}},  // Expected values
        // custom tests
        // 4d
        {{2, 3, 1, 2},                              // Input shape
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},   // Input values
         {1, 2, 0, 5},                              // Shift
         {10, 9, 12, 11, 8, 7, 4, 3, 6, 5, 2, 1}},  // Expected values
        // 5d
        {{1, 1, 3, 3, 2},                                                   // Input shape
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},   // Input values
         {1, 2, 10, 23, 6},                                                 // Shift
         {15, 16, 17, 18, 13, 14, 3, 4, 5, 6, 1, 2, 9, 10, 11, 12, 7, 8}},  // Expected values
        // 6d
        {{2, 1, 1, 3, 2, 3},                                                       // Input shape
         {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,  //
          19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36},     // Input values
         {0, 58, 27, 9, 99, 13},                                                   // Shift
         {6,  4,  5,  3,  1,  2,  12, 10, 11, 9,  7,  8,  18, 16, 17, 15, 13, 14,
          24, 22, 23, 21, 19, 20, 30, 28, 29, 27, 25, 26, 0,  35, 36, 34, 32, 33}},  // Expected values
    };
}

template <class T>
std::vector<roll_test_params<T>> getRollFloatingPointParams() {
    return {
        // from reference tests
        {{4, 3, 1, 1},  // Input shape
         {50.2907f,
          70.8054f,
          -68.3403f,
          62.6444f,
          4.9748f,
          -18.5551f,
          40.5383f,
          -15.3859f,
          -4.5881f,
          -43.3479f,
          94.1676f,
          -95.7097f},   // Input values
         {1, 0, 0, 0},  // Shift
         {{-43.3479f,
           94.1676f,
           -95.7097f,
           50.2907f,
           70.8054f,
           -68.3403f,
           62.6444f,
           4.9748f,
           -18.5551f,
           40.5383f,
           -15.3859f,
           -4.5881f}}},  // Expected values
        {{4, 3, 1, 1},   // Input shape
         {50.2907f,
          70.8054f,
          -68.3403f,
          62.6444f,
          4.9748f,
          -18.5551f,
          40.5383f,
          -15.3859f,
          -4.5881f,
          -43.3479f,
          94.1676f,
          -95.7097f},   // Input values
         {3, 2, 0, 0},  // Shift
         {{4.9748f,
           -18.5551f,
           62.6444f,
           -15.3859f,
           -4.5881f,
           40.5383f,
           94.1676f,
           -95.7097f,
           -43.3479f,
           70.8054f,
           -68.3403f,
           50.2907f}}},  // Expected values
        {{4, 2, 3, 1},   // Input shape
         {94.0773f,  33.0599f, 58.1724f,  -20.3640f, 54.5372f, -54.3023f, 10.4662f, 11.7532f,
          -11.7692f, 56.4223f, -95.3774f, 8.8978f,   1.9305f,  13.8025f,  12.0827f, 81.4669f,
          19.5321f,  -8.9553f, -75.3226f, 20.8033f,  20.7660f, 62.7361f,  14.9372f, -33.0825f},  // Input values
         {2, 1, 3, 0},                                                                           // Shift
         {{81.4669f,  19.5321f,  -8.9553f, 1.9305f,   13.8025f,  12.0827f, 62.7361f,  14.9372f,
           -33.0825f, -75.3226f, 20.8033f, 20.7660f,  -20.3640f, 54.5372f, -54.3023f, 94.0773f,
           33.0599f,  58.1724f,  56.4223f, -95.3774f, 8.8978f,   10.4662f, 11.7532f,  -11.7692f}}},  // Expected values
    };
}

#define INSTANTIATE_ROLL_TEST_SUITE(type, func)               \
    using roll_test_##type = roll_test<type>;                 \
    TEST_P(roll_test_##type, roll_##type) {                   \
        test();                                               \
    }                                                         \
    INSTANTIATE_TEST_SUITE_P(roll_smoke_##type,               \
                             roll_test_##type,                \
                             testing::ValuesIn(func<type>()), \
                             roll_test_##type::PrintToStringParamName);

INSTANTIATE_ROLL_TEST_SUITE(int8_t, getRollParams)
INSTANTIATE_ROLL_TEST_SUITE(uint8_t, getRollParams)
INSTANTIATE_ROLL_TEST_SUITE(int32_t, getRollParams)
INSTANTIATE_ROLL_TEST_SUITE(int64_t, getRollParams)
INSTANTIATE_ROLL_TEST_SUITE(FLOAT16, getRollFloatingPointParams)
INSTANTIATE_ROLL_TEST_SUITE(float, getRollFloatingPointParams)

#undef INSTANTIATE_ROLL_TEST_SUITE

}  // namespace
