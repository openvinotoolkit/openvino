// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reverse.hpp>
#include <string>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

template <reverse_mode mode>
struct ReverseModeTraits;

template <>
struct ReverseModeTraits<reverse_mode::index> {
    using axis_type = int32_t;
    static const data_types data_type = data_types::i32;
};

template <>
struct ReverseModeTraits<reverse_mode::mask> {
    using axis_type = bool;
    static const data_types data_type = data_types::u8;
};

/**
 * Specific Reverse params to define the tests. Input and output should be the same type
 */
template <typename T, reverse_mode mode>
struct ReverseParams {
    tensor input_tensor;
    format input_format;
    std::vector<T> input;
    std::vector<typename ReverseModeTraits<mode>::axis_type> axis;
    std::vector<T> expected_out;
};

template <typename T, reverse_mode mode>
struct reverse_gpu_test : public ::testing::TestWithParam<ReverseParams<T, mode>> {
public:
    void test() {
        auto data_type = type_to_data_type<T>::value;
        ReverseParams<T, mode> params = testing::TestWithParam<ReverseParams<T, mode>>::GetParam();
        auto& engine = get_test_engine();
        tensor t;

        auto reverse_input = engine.allocate_memory({data_type, params.input_format, params.input_tensor});
        auto reverse_axes = engine.allocate_memory(
            {ReverseModeTraits<mode>::data_type, format::bfyx, tensor(batch(1), feature(params.axis.size()))});

        set_values(reverse_input, params.input);
        set_values(reverse_axes, params.axis);

        const std::string reverse_id = "reverse";
        const std::string reverse_input_id = "reverse_input";
        const std::string axes_id = "reverse_axes";
        topology topology;
        topology.add(input_layout(reverse_input_id, reverse_input->get_layout()));
        topology.add(input_layout(axes_id, reverse_axes->get_layout()));

        topology.add(reverse(reverse_id, reverse_input_id, axes_id, mode));

        network network(engine, topology);

        network.set_input_data(reverse_input_id, reverse_input);
        network.set_input_data(axes_id, reverse_axes);

        auto result = network.execute();
        auto out_mem = result.at(reverse_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.expected_out.size(), out_ptr.size());
        for (size_t i = 0; i < params.expected_out.size(); ++i) {
            EXPECT_NEAR(params.expected_out[i], out_ptr[i], 0.0001) << "at i = " << i;
        }
    }
};

struct PrintToStringParamName {
    template <class T, reverse_mode mode>
    std::string operator()(const testing::TestParamInfo<ReverseParams<T, mode>>& param) {
        std::stringstream buf;
        buf << "input tensor " << param.param.input_tensor.to_string();
        buf << " axes {";
        for (auto val : param.param.axis) {
            buf << val << ",";
        }
        buf << "} format" << param.param.input_format.to_string();
        return buf.str();
    }
};

using reverse_gpu_test_int32_mask = reverse_gpu_test<int32_t, reverse_mode::mask>;
using reverse_gpu_test_int32_index = reverse_gpu_test<int32_t, reverse_mode::index>;
using reverse_gpu_test_int64_mask = reverse_gpu_test<int64_t, reverse_mode::mask>;
using reverse_gpu_test_int64_index = reverse_gpu_test<int64_t, reverse_mode::index>;
using reverse_gpu_test_float_mask = reverse_gpu_test<float, reverse_mode::mask>;
using reverse_gpu_test_float_index = reverse_gpu_test<float, reverse_mode::index>;
using reverse_gpu_test_int8_mask = reverse_gpu_test<int8_t, reverse_mode::mask>;
using reverse_gpu_test_int8_index = reverse_gpu_test<int8_t, reverse_mode::index>;
using reverse_gpu_test_uint8_mask = reverse_gpu_test<uint8_t, reverse_mode::mask>;
using reverse_gpu_test_uint8_index = reverse_gpu_test<uint8_t, reverse_mode::index>;
using reverse_gpu_test_f16_mask = reverse_gpu_test<half_t, reverse_mode::mask>;
using reverse_gpu_test_f16_index = reverse_gpu_test<half_t, reverse_mode::index>;

TEST_P(reverse_gpu_test_int32_mask, reverse_i32_mask) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_int32_index, reverse_i32_index) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_int64_mask, reverse_i64_mask) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_int64_index, reverse_i64_index) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_float_mask, reverse_float_mask) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_float_index, reverse_float_index) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_int8_mask, reverse_int8_mask) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_int8_index, reverse_int8_index) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_uint8_mask, reverse_uint8_mask) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_uint8_index, reverse_uint8_index) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_f16_mask, reverse_f16_mask) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(reverse_gpu_test_f16_index, reverse_f16_index) {
    ASSERT_NO_FATAL_FAILURE(test());
}

template <typename T>
std::vector<ReverseParams<T, reverse_mode::mask>> generateMaskParams() {
    std::vector<ReverseParams<T, reverse_mode::mask>> params{
        // reverse_2d_1_mask
        {tensor(batch(4), feature(3)),
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
         {false, true},
         std::vector<T>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}},
        {tensor(batch(4), feature(3)),
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
         {true, true},
         std::vector<T>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
        {tensor(batch(4), feature(3)),
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
         {false, false},
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
    };
    return params;
}

template <typename T>
std::vector<ReverseParams<T, reverse_mode::index>> generateIndexParams() {
    std::vector<ReverseParams<T, reverse_mode::index>> params{

        //{tensor(batch(8)), format::bfyx, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7}, {},
        // std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7}},
        {tensor(batch(8)),
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
         {0},
         std::vector<T>{7, 6, 5, 4, 3, 2, 1, 0}},
        {tensor(batch(4), feature(3)),
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
         {0},
         std::vector<T>{9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2}},
        {tensor(batch(4), feature(3)),
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
         {1},
         std::vector<T>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}},
        {tensor(batch(4), feature(3)),
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
         {0, 1},
         std::vector<T>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
        {tensor{1, 1, 3, 4, 2},
         format::bfzyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
         {2},
         std::vector<T>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
        {tensor{1, 1, 3, 4, 2},
         format::bfzyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
         {3},
         std::vector<T>{9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14}},
        {tensor{1, 1, 3, 4, 2},
         format::bfzyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
         {4},
         std::vector<T>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21}},
        {tensor{2, 4, 1, 3},
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
         {0, 1},
         std::vector<T>{21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2}},
        {tensor{
             2,
             4,
             1,
             3,
         },
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
         {0, 2},
         std::vector<T>{14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21, 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}},
        {tensor{2, 4, 1, 3},
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
         {1, 2},
         std::vector<T>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12}},
        {tensor{2, 4, 1, 3},
         format::bfyx,
         std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
         {0, 1, 2},
         std::vector<T>{23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}}};
    return params;
}

template <>
std::vector<ReverseParams<half_t, reverse_mode::mask>> generateMaskParams() {
    std::vector<ReverseParams<half_t, reverse_mode::mask>> params{// reverse_2d_1_mask
                                                                  {tensor(batch(4), feature(3)),
                                                                   format::bfyx,
                                                                   std::vector<half_t>{half_t(0),
                                                                                       half_t(1),
                                                                                       half_t(2),
                                                                                       half_t(3),
                                                                                       half_t(4),
                                                                                       half_t(5),
                                                                                       half_t(6),
                                                                                       half_t(7),
                                                                                       half_t(8),
                                                                                       half_t(9),
                                                                                       half_t(10),
                                                                                       half_t(11)},
                                                                   {false, true},
                                                                   std::vector<half_t>{half_t(2),
                                                                                       half_t(1),
                                                                                       half_t(0),
                                                                                       half_t(5),
                                                                                       half_t(4),
                                                                                       half_t(3),
                                                                                       half_t(8),
                                                                                       half_t(7),
                                                                                       half_t(6),
                                                                                       half_t(11),
                                                                                       half_t(10),
                                                                                       half_t(9)}}};
    return params;
}

template <>
std::vector<ReverseParams<half_t, reverse_mode::index>> generateIndexParams() {
    std::vector<ReverseParams<half_t, reverse_mode::index>> params{// reverse_2d_1_mask
                                                                   {tensor(batch(4), feature(3)),
                                                                    format::bfyx,
                                                                    std::vector<half_t>{half_t(0),
                                                                                        half_t(1),
                                                                                        half_t(2),
                                                                                        half_t(3),
                                                                                        half_t(4),
                                                                                        half_t(5),
                                                                                        half_t(6),
                                                                                        half_t(7),
                                                                                        half_t(8),
                                                                                        half_t(9),
                                                                                        half_t(10),
                                                                                        half_t(11)},
                                                                    {1},
                                                                    std::vector<half_t>{half_t(2),
                                                                                        half_t(1),
                                                                                        half_t(0),
                                                                                        half_t(5),
                                                                                        half_t(4),
                                                                                        half_t(3),
                                                                                        half_t(8),
                                                                                        half_t(7),
                                                                                        half_t(6),
                                                                                        half_t(11),
                                                                                        half_t(10),
                                                                                        half_t(9)}}};
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_reverse_i32_mask,
                         reverse_gpu_test_int32_mask,
                         ::testing::ValuesIn(generateMaskParams<int32_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_i64_mask,
                         reverse_gpu_test_int64_mask,
                         ::testing::ValuesIn(generateMaskParams<int64_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_float_mask,
                         reverse_gpu_test_float_mask,
                         ::testing::ValuesIn(generateMaskParams<float>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_int8_mask,
                         reverse_gpu_test_int8_mask,
                         ::testing::ValuesIn(generateMaskParams<int8_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_uint8_mask,
                         reverse_gpu_test_uint8_mask,
                         ::testing::ValuesIn(generateMaskParams<uint8_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_f16_mask,
                         reverse_gpu_test_f16_mask,
                         ::testing::ValuesIn(generateMaskParams<half_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_i32_index,
                         reverse_gpu_test_int32_index,
                         ::testing::ValuesIn(generateIndexParams<int32_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_i64_index,
                         reverse_gpu_test_int64_index,
                         ::testing::ValuesIn(generateIndexParams<int64_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_float_index,
                         reverse_gpu_test_float_index,
                         ::testing::ValuesIn(generateIndexParams<float>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_int8_index,
                         reverse_gpu_test_int8_index,
                         ::testing::ValuesIn(generateIndexParams<int8_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_uint8_index,
                         reverse_gpu_test_uint8_index,
                         ::testing::ValuesIn(generateIndexParams<uint8_t>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_reverse_f16_index,
                         reverse_gpu_test_f16_index,
                         ::testing::ValuesIn(generateIndexParams<half_t>()),
                         PrintToStringParamName());
