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
    void test(bool is_caching_test = false) {
        auto data_type = ov::element::from<T>();
        ReverseParams<T, mode> params = testing::TestWithParam<ReverseParams<T, mode>>::GetParam();
        auto& engine = get_test_engine();

        format fmt = generic_test::get_plain_format_for(params.input_format);
        bool reorder_needed = fmt != params.input_format;

        auto reverse_input = engine.allocate_memory({data_type, fmt, params.input_tensor});
        auto reverse_axes = engine.allocate_memory(
            {ReverseModeTraits<mode>::data_type, fmt, tensor(batch(1), feature(params.axis.size()))});
        set_values(reverse_input, params.input);
        set_values(reverse_axes, params.axis);

        const std::string reverse_input_id = "reverse_input";
        const std::string axes_id = "reverse_axes";
        topology tp;
        tp.add(input_layout(reverse_input_id, reverse_input->get_layout()));
        tp.add(input_layout(axes_id, reverse_axes->get_layout()));
        const std::string reverse_id = "reverse";
        std::string ouput_op_name{reverse_id};
        if (reorder_needed) {
            const std::string r_reverse_input_id = "r_reverse_input";
            const std::string r_axes_id = "r_reverse_axes";
            tp.add(reorder(r_reverse_input_id, input_info(reverse_input_id), params.input_format, ov::element::from<T>()));
            tp.add(reorder(r_axes_id, input_info(axes_id), params.input_format, ov::element::from<T>()));
            tp.add(reverse(reverse_id, input_info(r_reverse_input_id), input_info(r_axes_id), mode));
            ouput_op_name = "reversed_result";
            tp.add(reorder(ouput_op_name, input_info(reverse_id), fmt, ov::element::from<T>()));
        } else {
            tp.add(reverse(reverse_id, input_info(reverse_input_id), input_info(axes_id), mode));
        }

        cldnn::network::ptr network = get_network(engine, tp, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        network->set_input_data(reverse_input_id, reverse_input);
        network->set_input_data(axes_id, reverse_axes);
        auto result = network->execute();

        auto out_mem = result.at(ouput_op_name).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.expected_out.size(), out_ptr.size());
        for (size_t i = 0; i < params.expected_out.size(); ++i) {
            ASSERT_NEAR(params.expected_out[i], out_ptr[i], 0.0001) << "at i = " << i;
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
        buf << "} format " << param.param.input_format.to_string();
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
using reverse_gpu_test_f16_mask = reverse_gpu_test<ov::float16, reverse_mode::mask>;
using reverse_gpu_test_f16_index = reverse_gpu_test<ov::float16, reverse_mode::index>;

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

namespace {

const auto four_d_formats = {
    format::bfyx,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv32,
    format::bs_fs_yx_bsv32_fsv16,
};

const auto five_d_formats = {
    format::bfzyx,
    format::b_fs_zyx_fsv16,
    format::b_fs_zyx_fsv32,
    format::bs_fs_zyx_bsv16_fsv32,
    format::bs_fs_zyx_bsv16_fsv16,
    format::bs_fs_zyx_bsv32_fsv32,
    format::bs_fs_zyx_bsv32_fsv16,
};
}  // namespace

template <typename T>
std::vector<ReverseParams<T, reverse_mode::mask>> generateMaskParams() {
    std::vector<ReverseParams<T, reverse_mode::mask>> params;
    for (const auto f : four_d_formats) {
        params.push_back({tensor(batch(4), feature(3)),
                          f,
                          std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                          {false, true},
                          std::vector<T>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}});
        params.push_back({tensor(batch(4), feature(3)),
                          f,
                          std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                          {true, true},
                          std::vector<T>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}});

        params.push_back({tensor(batch(4), feature(3)),
                          f,
                          std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                          {false, false},
                          std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}});
    }
    return params;
}

template <typename T>
std::vector<ReverseParams<T, reverse_mode::index>> generateIndexParams() {
    std::vector<ReverseParams<T, reverse_mode::index>> params;
    for (const auto fmt : four_d_formats) {
        std::vector<ReverseParams<T, reverse_mode::index>> local_params{
            //{tensor(batch(8)), format::bfyx, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7}, {},
            // std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7}},
            {tensor(batch(8)),
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7},
             {0},
             std::vector<T>{7, 6, 5, 4, 3, 2, 1, 0}},
            {tensor(batch(4), feature(3)),
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
             {0},
             std::vector<T>{9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2}},
            {tensor(batch(4), feature(3)),
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
             {1},
             std::vector<T>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}},
            {tensor(batch(4), feature(3)),
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
             {0, 1},
             std::vector<T>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
            {tensor{2, 4, 1, 3},
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
             {0, 1},
             std::vector<T>{21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2}},
            {tensor{
                 2,
                 4,
                 1,
                 3,
             },
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
             {0, 2},
             std::vector<T>{14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21, 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}},
            {tensor{2, 4, 1, 3},
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
             {1, 2},
             std::vector<T>{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12}},
            {tensor{2, 4, 1, 3},
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
             {0, 1, 2},
             std::vector<T>{23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}}};
        std::move(local_params.begin(), local_params.end(), std::back_inserter(params));
    }

    for (const auto fmt : five_d_formats) {
        std::vector<ReverseParams<T, reverse_mode::index>> local_params{
            {tensor{1, 1, 3, 4, 2},
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
             {2},
             std::vector<T>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
            {tensor{1, 1, 3, 4, 2},
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
             {3},
             std::vector<T>{9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14}},
            {tensor{1, 1, 3, 4, 2},
             fmt,
             std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
             {4},
             std::vector<T>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21}},
        };
        std::move(local_params.begin(), local_params.end(), std::back_inserter(params));
    }
    return params;
}

template <>
std::vector<ReverseParams<ov::float16, reverse_mode::mask>> generateMaskParams() {
    std::vector<ReverseParams<ov::float16, reverse_mode::mask>> params;
    for (const auto fmt : four_d_formats) {
        // reverse_2d_1_mask
        params.push_back({tensor(batch(4), feature(3)),
                          fmt,
                          std::vector<ov::float16>{ov::float16(0),
                                              ov::float16(1),
                                              ov::float16(2),
                                              ov::float16(3),
                                              ov::float16(4),
                                              ov::float16(5),
                                              ov::float16(6),
                                              ov::float16(7),
                                              ov::float16(8),
                                              ov::float16(9),
                                              ov::float16(10),
                                              ov::float16(11)},
                          {false, true},
                          std::vector<ov::float16>{ov::float16(2),
                                              ov::float16(1),
                                              ov::float16(0),
                                              ov::float16(5),
                                              ov::float16(4),
                                              ov::float16(3),
                                              ov::float16(8),
                                              ov::float16(7),
                                              ov::float16(6),
                                              ov::float16(11),
                                              ov::float16(10),
                                              ov::float16(9)}});
    }

    return params;
}

template <>
std::vector<ReverseParams<ov::float16, reverse_mode::index>> generateIndexParams() {
    std::vector<ReverseParams<ov::float16, reverse_mode::index>> params;
    for (const auto fmt : four_d_formats) {
        // reverse_2d_1_mask
        params.push_back({tensor(batch(4), feature(3)),
                          fmt,
                          std::vector<ov::float16>{ov::float16(0),
                                              ov::float16(1),
                                              ov::float16(2),
                                              ov::float16(3),
                                              ov::float16(4),
                                              ov::float16(5),
                                              ov::float16(6),
                                              ov::float16(7),
                                              ov::float16(8),
                                              ov::float16(9),
                                              ov::float16(10),
                                              ov::float16(11)},
                          {1},
                          std::vector<ov::float16>{ov::float16(2),
                                              ov::float16(1),
                                              ov::float16(0),
                                              ov::float16(5),
                                              ov::float16(4),
                                              ov::float16(3),
                                              ov::float16(8),
                                              ov::float16(7),
                                              ov::float16(6),
                                              ov::float16(11),
                                              ov::float16(10),
                                              ov::float16(9)}});
    }
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
                         ::testing::ValuesIn(generateMaskParams<ov::float16>()),
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
                         ::testing::ValuesIn(generateIndexParams<ov::float16>()),
                         PrintToStringParamName());

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(reverse_gpu_test_int32_mask, reverse_i32_mask_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_int32_index, reverse_i32_index_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_int64_mask, reverse_i64_mask_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_int64_index, reverse_i64_index_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_float_mask, reverse_float_mask_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_float_index, reverse_float_index_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_int8_mask, reverse_int8_mask_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_int8_index, reverse_int8_index_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_uint8_mask, reverse_uint8_mask_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_uint8_index, reverse_uint8_index_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(reverse_gpu_test_f16_mask, reverse_f16_mask_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}
#endif
TEST_P(reverse_gpu_test_f16_index, reverse_f16_index_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}
