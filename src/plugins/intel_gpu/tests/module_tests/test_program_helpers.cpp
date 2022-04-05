// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include "program_helpers.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include <intel_gpu/primitives/data.hpp>

#include <cmath>
#include <limits>

using namespace cldnn;
using namespace ::tests;
using namespace testing;

struct test_param {
    tensor in_shape;
    tensor out_shape;
    format input_format;
    format output_format;
    padding pad1;
    padding pad2;
    data_types input_data_type;
    data_types output_data_type;
};

template<typename T>
class ReorderTest : public ::testing::TestWithParam<T> {};

class functional_test : public ReorderTest<test_param> {};
TEST_P(functional_test, test_are_layouts_identical) {
    auto p = GetParam();

    layout in_layout(p.input_data_type, p.input_format, p.in_shape, p.pad1);
    layout in_layout_no_pad(p.input_data_type, p.input_format, p.in_shape, p.pad2);
    layout in_bfyx_layout(p.input_data_type, format::bfyx, p.in_shape, p.pad2);
    layout out_layout(p.output_data_type, p.output_format, p.out_shape, p.pad1);
    layout out_layout_no_pad(p.output_data_type, p.output_format, p.out_shape, p.pad2);
    layout out_bfyx_layout(p.output_data_type, format::bfyx, p.out_shape, p.pad2);

    auto test1 = program_helpers::are_layouts_identical(in_layout, out_layout);
    EXPECT_EQ(true, test1.first);
    EXPECT_EQ(true, test1.second);
    auto test2 = program_helpers::are_layouts_identical(in_layout, layout(data_types::f32, p.output_format, p.out_shape, p.pad1));
    EXPECT_EQ(false, test2.first);
    EXPECT_EQ(false, test2.second);
    auto test3 = program_helpers::are_layouts_identical(in_bfyx_layout, out_bfyx_layout);
    EXPECT_EQ(true, test3.first);
    EXPECT_EQ(true, test3.second);
    auto test4 = program_helpers::are_layouts_identical(in_bfyx_layout, layout(p.input_data_type, format::bfzyx, p.in_shape, p.pad2));
    EXPECT_EQ(false, test4.first);
    EXPECT_EQ(true, test4.second);
    auto test5 = program_helpers::are_layouts_identical(in_bfyx_layout, layout(p.input_data_type, format::bfzyx, p.in_shape, p.pad1));
    EXPECT_EQ(false, test5.first);
    EXPECT_EQ(false, test5.second);
    auto test6 = program_helpers::are_layouts_identical(in_layout, layout(p.input_data_type, p.input_format, {1, 32, 16, 16}, p.pad1));
    EXPECT_EQ(false, test6.first);
    EXPECT_EQ(false, test6.second);
    auto test7 = program_helpers::are_layouts_identical(in_bfyx_layout, layout(p.input_data_type, format::b_fs_yx_fsv32, p.in_shape, p.pad2));
    EXPECT_EQ(false, test7.first);
    EXPECT_EQ(false, test7.second);
    auto test8 = program_helpers::are_layouts_identical(layout(p.input_data_type, format::b_fs_yx_fsv16, p.in_shape, p.pad2), in_bfyx_layout);
    EXPECT_EQ(false, test8.first);
    EXPECT_EQ(false, test8.second);
    auto test9 = program_helpers::are_layouts_identical(in_layout, layout(p.input_data_type, p.input_format, p.in_shape, p.pad2));
    EXPECT_EQ(false, test9.first);
    EXPECT_EQ(false, test9.second);
    tensor temp = p.in_shape;
    temp = temp.sub(p.pad1.lower_size());
    auto test10 = program_helpers::are_layouts_identical(in_bfyx_layout, layout(p.input_data_type, format::bfzyx, temp, p.pad1));
    EXPECT_EQ(false, test10.first);
    EXPECT_EQ(false, test10.second);
}

INSTANTIATE_TEST_SUITE_P(same_in_out,
                        functional_test,
                        ::testing::ValuesIn(std::vector<test_param>{
                                            test_param{{1, 32, 4, 4}, {1, 32, 4, 4}, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, padding({0, 0, 1, 1}, 0), padding({0, 0, 0, 0}, 0), data_types::f16, data_types::f16},
                                            test_param{{1, 32, 4, 4}, {1, 32, 4, 4}, format::bfyx, format::bfyx, padding({0, 0, 1, 1}, 0), padding({0, 0, 0, 0}, 0), data_types::u8, data_types::u8}
                                            }));
