// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "intel_gpu/primitives/roll.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace tests;

namespace {

template <class T>
struct roll_test_input {
    std::vector<ov::Dimension::value_type> input_shape;
    std::vector<T> input_values;
    std::vector<ov::Dimension::value_type> shift;
    std::vector<T> expected_values;
};

template <class T>
using roll_test_params = std::tuple<roll_test_input<T>, format::type>;

template <class T>
struct roll_test : testing::TestWithParam<roll_test_params<T>> {
    void test(bool is_caching_test) {
        roll_test_input<T> p;
        format::type input_format;
        std::tie(p, input_format) = testing::TestWithParam<roll_test_params<T>>::GetParam();
        auto& engine = get_test_engine();

        format::type plane_format = format::get_default_format(p.input_shape.size());
        const layout data_layout(ov::element::from<T>(), plane_format, tensor(input_format, p.input_shape));
        auto input = engine.allocate_memory(data_layout);
        set_values(input, p.input_values);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("reordered_input", input_info("input"), input_format, ov::element::from<T>()));
        topology.add(roll("roll", input_info("reordered_input"), tensor(input_format, p.shift)));
        topology.add(reorder("reordered_roll", input_info("roll"), plane_format, ov::element::from<T>()));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        const auto outputs = network->execute();

        auto output = outputs.at("reordered_roll").get_memory();
        cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), p.expected_values.size());
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            ASSERT_NEAR(p.expected_values[i], output_ptr[i], 1e-5f);
        }
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<roll_test_params<T>>& info) {
        auto& p = std::get<0>(info.param);
        std::ostringstream result;
        result << "InputShape=" << vec2str(p.input_shape) << "_";
        result << "Precision=" << ov::element::Type(ov::element::from<T>()) << "_";
        result << "Shift=" << vec2str(p.shift) << "_";
        result << "Format=" << std::get<1>(info.param);
        return result.str();
    }
};

template <class T>
std::vector<roll_test_input<T>> getRollParamsToCheckLogic() {
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
    };
}

template <class T>
std::vector<roll_test_input<T>> getRollParamsToCheckLayouts() {
    return {
        // custom tests
        // 4d
        {{2, 3, 1, 2},                              // Input shape
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},   // Input values
         {1, 2, 0, 5},                              // Shift
         {10, 9, 12, 11, 8, 7, 4, 3, 6, 5, 2, 1}},  // Expected values
    };
}

template <class T>
std::vector<roll_test_input<T>> getRollParams5D() {
    return {
        // 5d
        {{1, 1, 3, 3, 2},                                                   // Input shape
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},   // Input values
         {1, 2, 10, 23, 6},                                                 // Shift
         {15, 16, 17, 18, 13, 14, 3, 4, 5, 6, 1, 2, 9, 10, 11, 12, 7, 8}},  // Expected values
    };
}

template <class T>
std::vector<roll_test_input<T>> getRollParams6D() {
    return {
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
std::vector<roll_test_input<T>> getRollFloatingPointParams() {
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
    };
}

template <class T>
std::vector<roll_test_input<T>> getRollFloatingPointAdditionalLogic() {
    return {
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

std::vector<format::type> formats4d = {format::bfyx,
                                       format::bs_fs_yx_bsv32_fsv32,
                                       format::bs_fs_yx_bsv32_fsv16,
                                       format::b_fs_yx_fsv32,
                                       format::b_fs_yx_fsv16,
                                       format::bs_fs_yx_bsv16_fsv16};

std::vector<format::type> formats5d = {format::bfzyx,
                                       format::bs_fs_zyx_bsv32_fsv32,
                                       format::bs_fs_zyx_bsv32_fsv16,
                                       format::b_fs_zyx_fsv32,
                                       format::b_fs_zyx_fsv16,
                                       format::bs_fs_zyx_bsv16_fsv16};

std::vector<format::type> formats6d = {format::bfwzyx};

#define INSTANTIATE_ROLL_TEST_SUITE(type, func, formats)                                                    \
    class roll_test_##type##func : public roll_test<type> {};                                               \
    TEST_P(roll_test_##type##func, roll_##type##func) {                                                     \
        test(false);                                                                                        \
    }                                                                                                       \
    INSTANTIATE_TEST_SUITE_P(roll_smoke_##type##func,                                                       \
                             roll_test_##type##func,                                                        \
                             testing::Combine(testing::ValuesIn(func<type>()), testing::ValuesIn(formats)), \
                             roll_test_##type##func::PrintToStringParamName);

INSTANTIATE_ROLL_TEST_SUITE(int8_t, getRollParamsToCheckLogic, {format::bfyx})
INSTANTIATE_ROLL_TEST_SUITE(uint8_t, getRollParamsToCheckLogic, {format::bfyx})
INSTANTIATE_ROLL_TEST_SUITE(int32_t, getRollParamsToCheckLogic, {format::bfyx})
INSTANTIATE_ROLL_TEST_SUITE(int64_t, getRollParamsToCheckLogic, {format::bfyx})
INSTANTIATE_ROLL_TEST_SUITE(int8_t, getRollParamsToCheckLayouts, formats4d)
INSTANTIATE_ROLL_TEST_SUITE(uint8_t, getRollParamsToCheckLayouts, formats4d)
INSTANTIATE_ROLL_TEST_SUITE(int32_t, getRollParamsToCheckLayouts, formats4d)
INSTANTIATE_ROLL_TEST_SUITE(int64_t, getRollParamsToCheckLayouts, formats4d)
INSTANTIATE_ROLL_TEST_SUITE(int8_t, getRollParams5D, formats5d)
INSTANTIATE_ROLL_TEST_SUITE(uint8_t, getRollParams5D, formats5d)
INSTANTIATE_ROLL_TEST_SUITE(int32_t, getRollParams5D, formats5d)
INSTANTIATE_ROLL_TEST_SUITE(int64_t, getRollParams5D, formats5d)
INSTANTIATE_ROLL_TEST_SUITE(int8_t, getRollParams6D, formats6d)
INSTANTIATE_ROLL_TEST_SUITE(uint8_t, getRollParams6D, formats6d)
INSTANTIATE_ROLL_TEST_SUITE(int32_t, getRollParams6D, formats6d)
INSTANTIATE_ROLL_TEST_SUITE(int64_t, getRollParams6D, formats6d)

using ov::float16;
INSTANTIATE_ROLL_TEST_SUITE(float16, getRollFloatingPointParams, formats4d)
INSTANTIATE_ROLL_TEST_SUITE(float, getRollFloatingPointParams, formats4d)
INSTANTIATE_ROLL_TEST_SUITE(float16, getRollFloatingPointAdditionalLogic, {format::bfyx})
INSTANTIATE_ROLL_TEST_SUITE(float, getRollFloatingPointAdditionalLogic, {format::bfyx})

#undef INSTANTIATE_ROLL_TEST_SUITE

#define INSTANTIATE_ROLL_TEST_SUITE_CACHED(type, func)           \
    TEST_P(roll_test_##type##func, roll_##type##func##_cached) { \
        test(true);                                              \
    }

#ifdef RUN_ALL_MODEL_CACHING_TESTS
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int8_t, getRollParamsToCheckLogic)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(uint8_t, getRollParamsToCheckLogic)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int32_t, getRollParamsToCheckLogic)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int64_t, getRollParamsToCheckLogic)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int8_t, getRollParamsToCheckLayouts)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(uint8_t, getRollParamsToCheckLayouts)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int32_t, getRollParamsToCheckLayouts)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int64_t, getRollParamsToCheckLayouts)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int8_t, getRollParams5D)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(uint8_t, getRollParams5D)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int32_t, getRollParams5D)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int64_t, getRollParams5D)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int8_t, getRollParams6D)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(uint8_t, getRollParams6D)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int32_t, getRollParams6D)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(int64_t, getRollParams6D)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(float16, getRollFloatingPointParams)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(float, getRollFloatingPointParams)
INSTANTIATE_ROLL_TEST_SUITE_CACHED(float16, getRollFloatingPointAdditionalLogic)
#endif
INSTANTIATE_ROLL_TEST_SUITE_CACHED(float, getRollFloatingPointAdditionalLogic)

#undef INSTANTIATE_ROLL_TEST_SUITE_CACHED
}  // namespace
