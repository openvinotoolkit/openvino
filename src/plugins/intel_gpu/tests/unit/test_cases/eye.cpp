// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <algorithm>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eye.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <random>
#include <vector>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

template <class OutputType, class InputType>
using eye_test_param = std::tuple<format,                    // Input and output format
                                  InputType,                 // columns number
                                  InputType,                 // rows number
                                  InputType,                 // diagonal index
                                  std::vector<InputType>,    // batch shape
                                  std::vector<ov::Dimension::value_type>,      // output shape
                                  std::vector<OutputType>,   // expected values
                                  bool>;                     // is_caching_test

template <class OutputType, class InputType>
class EyeTest : public ::testing::TestWithParam<eye_test_param<OutputType, InputType>> {
public:
    void SetUp() override {
        format fmt{format::bfyx};
        InputType cols{};
        InputType rows{};
        InputType diag{};
        std::vector<InputType> batch_shape;
        std::vector<ov::Dimension::value_type> output_shape;
        std::vector<OutputType> expected_values;
        bool is_caching_test;
        std::tie(fmt, cols, rows, diag, batch_shape, output_shape, expected_values, is_caching_test) = this->GetParam();

        auto num_rows = engine_.allocate_memory({ov::element::from<InputType>(), fmt, tensor{1}});
        set_values<InputType>(num_rows, {rows});
        auto num_coloms = engine_.allocate_memory({ov::element::from<InputType>(), fmt, tensor{1}});
        set_values<InputType>(num_coloms, {cols});
        auto diagonal_index = engine_.allocate_memory({ov::element::from<InputType>(), fmt, tensor{1}});
        set_values<InputType>(diagonal_index, {diag});

        topology tp;
        tp.add(data("num_rows", num_rows));
        tp.add(data("num_columns", num_coloms));
        tp.add(data("diagonal_index", diagonal_index));

        auto batch_rank = batch_shape.size() == 3 ? 3 : 2;
        auto oupput_fmt = batch_rank == 3 ? format::bfzyx : format::bfyx;
        if (!batch_shape.empty()) {
            auto batch = engine_.allocate_memory({ov::element::from<InputType>(), fmt, tensor{batch_rank}});
            set_values<InputType>(batch, batch_shape);
            tp.add(data("batch", batch));
        }

        std::string ouput_op_name;
        if (fmt == format::bfyx || fmt == format::bfzyx) {
            auto inputs = batch_shape.empty()
                              ? std::vector<input_info>{ input_info("num_rows"), input_info("num_columns"), input_info("diagonal_index") }
                              : std::vector<input_info>{ input_info("num_rows"), input_info("num_columns"), input_info("diagonal_index"), input_info("batch") };
            ouput_op_name = "eye";
            auto eye_primitive =
                eye("eye", inputs, tensor{output_shape}, diag, ov::element::from<OutputType>());
            tp.add(std::move(eye_primitive));
        } else {
            tp.add(reorder("r_num_rows", input_info("num_rows"), fmt, ov::element::from<InputType>()));
            tp.add(reorder("r_num_columns", input_info("num_columns"), fmt, ov::element::from<InputType>()));
            tp.add(reorder("r_diagonal_index", input_info("diagonal_index"), fmt, ov::element::from<InputType>()));
            if (!batch_shape.empty()) {
                tp.add(reorder("r_batch", input_info("batch"), fmt, ov::element::from<InputType>()));
            }
            auto inputs = batch_shape.empty()
                              ? std::vector<input_info>{ input_info("r_num_rows"), input_info("r_num_columns"), input_info("r_diagonal_index") }
                              : std::vector<input_info>{ input_info("r_num_rows"), input_info("r_num_columns"), input_info("r_diagonal_index"), input_info("r_batch") };
            auto eye_primitive =
                eye("eye", inputs, tensor{output_shape}, diag, ov::element::from<OutputType>());
            tp.add(std::move(eye_primitive));
            ouput_op_name = "output";
            tp.add(reorder("output", input_info("eye"), oupput_fmt, ov::element::from<OutputType>()));
        }

        cldnn::network::ptr network = get_network(engine_, tp, get_test_default_config(engine_), get_test_stream_ptr(), is_caching_test);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, ouput_op_name);

        auto output = outputs.at(ouput_op_name).get_memory();

        cldnn::mem_lock<OutputType> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), expected_values.size());
        for (size_t i = 0; i < output_ptr.size(); ++i)
            ASSERT_TRUE(are_equal(expected_values[i], output_ptr[i], 2e-3));
    }

protected:
    engine& engine_ = get_test_engine();
};

std::vector<format> four_d_formats{
    format::bfyx,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv32,
    format::bs_fs_yx_bsv32_fsv16,
};

using eye_test_4d_float_int32 = EyeTest<float, int32_t>;
TEST_P(eye_test_4d_float_int32, eye_test_4d_float_int32) {}
INSTANTIATE_TEST_SUITE_P(
    eye_test_4d_float_int32,
    eye_test_4d_float_int32,
    testing::Combine(testing::ValuesIn(four_d_formats),
                     testing::Values(2),
                     testing::Values(3),
                     testing::Values(0),
                     testing::ValuesIn(std::vector<std::vector<int32_t>>{{}, {1}, {1, 1}, {1, 1, 1}}),
                     testing::Values(std::vector<ov::Dimension::value_type>{1, 1, 2, 3}),
                     testing::Values(std::vector<float>{1, 0, 0, 1, 0, 0}),
                     testing::Values(false)));

using eye_test_4d_int64_int32 = EyeTest<int64_t, int32_t>;
TEST_P(eye_test_4d_int64_int32, eye_test_4d_int64_int32) {}
INSTANTIATE_TEST_SUITE_P(
    eye_test_4d_int64_int32,
    eye_test_4d_int64_int32,
    testing::Combine(testing::ValuesIn(four_d_formats),
                     testing::Values(2),
                     testing::Values(3),
                     testing::Values(0),
                     testing::ValuesIn(std::vector<std::vector<int32_t>>{{}, {1}, {1, 1}, {1, 1, 1}}),
                     testing::Values(std::vector<ov::Dimension::value_type>{1, 1, 2, 3}),
                     testing::Values(std::vector<int64_t>{1, 0, 0, 1, 0, 0}),
                     testing::Values(false)));

using eye_test_4d_u8_int64 = EyeTest<uint8_t, int64_t>;
TEST_P(eye_test_4d_u8_int64, eye_test_4d_u8_int64) {}
INSTANTIATE_TEST_SUITE_P(
    eye_test_4d_u8_int64,
    eye_test_4d_u8_int64,
    testing::Combine(testing::ValuesIn(four_d_formats),
                     testing::Values(4),
                     testing::Values(3),
                     testing::Values(-1),
                     testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {1}, {1, 1}, {1, 1, 1}}),
                     testing::Values(std::vector<ov::Dimension::value_type>{1, 1, 4, 3}),
                     testing::Values(std::vector<uint8_t>{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
                     testing::Values(false)));

using eye_test_4d_i8_int64_no_diag = EyeTest<int8_t, int64_t>;
TEST_P(eye_test_4d_i8_int64_no_diag, eye_test_4d_i8_int64_no_diag) {}
INSTANTIATE_TEST_SUITE_P(
    eye_test_4d_i8_int64_no_diag,
    eye_test_4d_i8_int64_no_diag,
    testing::Combine(testing::ValuesIn(four_d_formats),
                     testing::Values(4),
                     testing::Values(3),
                     testing::Values(4),
                     testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {1}, {1, 1}, {1, 1, 1}}),
                     testing::Values(std::vector<ov::Dimension::value_type>{1, 1, 4, 3}),
                     testing::Values(std::vector<int8_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                     testing::Values(false)));

using eye_test_4d_int32_int32_batch = EyeTest<int32_t, int32_t>;
TEST_P(eye_test_4d_int32_int32_batch, eye_test_4d_int32_int32_batch) {}
INSTANTIATE_TEST_SUITE_P(
    eye_test_4d_int32_int32_batch,
    eye_test_4d_int32_int32_batch,
    testing::Combine(testing::ValuesIn(four_d_formats),
                     testing::Values(2),
                     testing::Values(2),
                     testing::Values(1),
                     testing::ValuesIn(std::vector<std::vector<int32_t>>{{2, 2}}),
                     testing::Values(std::vector<ov::Dimension::value_type>{2, 2, 2, 2}),
                     testing::Values(std::vector<int32_t>{0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}),
                     testing::Values(false)));

std::vector<format> five_d_formats{
    format::bfzyx,
    format::b_fs_zyx_fsv16,
    format::b_fs_zyx_fsv32,
    format::bs_fs_zyx_bsv16_fsv32,
    format::bs_fs_zyx_bsv16_fsv16,
    format::bs_fs_zyx_bsv32_fsv32,
    format::bs_fs_zyx_bsv32_fsv16,
};

using eye_test_5d_float_int32 = EyeTest<float, int32_t>;
TEST_P(eye_test_5d_float_int32, eye_test_5d_float_int32) {}
INSTANTIATE_TEST_SUITE_P(eye_test_5d_float_int32,
                         eye_test_5d_float_int32,
                         testing::Combine(testing::Values(five_d_formats[0]),
                                          testing::Values(2),
                                          testing::Values(2),
                                          testing::Values(0),
                                          testing::ValuesIn(std::vector<std::vector<int32_t>>{{2, 2, 2}}),
                                          testing::Values(std::vector<ov::Dimension::value_type>{2, 2, 2, 2, 2}),
                                          testing::Values(std::vector<float>{
                                              1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,

                                              1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                                          }),
                                          testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(export_import,
                         eye_test_5d_float_int32,
                         testing::Combine(testing::Values(five_d_formats[0]),
                                          testing::Values(2),
                                          testing::Values(2),
                                          testing::Values(0),
                                          testing::ValuesIn(std::vector<std::vector<int32_t>>{{2, 2, 2}}),
                                          testing::Values(std::vector<ov::Dimension::value_type>{2, 2, 2, 2, 2}),
                                          testing::Values(std::vector<float>{
                                              1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,

                                              1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                                          }),
                                          testing::Values(true)));

}  // anonymous namespace
