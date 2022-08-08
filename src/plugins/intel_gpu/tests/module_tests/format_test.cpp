// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/format.hpp"

using namespace cldnn;

TEST(format, to_string) {
    typedef std::underlying_type<cldnn::format::type>::type format_underlying_type;
    for (format_underlying_type i = 0; i < static_cast<format_underlying_type>(cldnn::format::format_num); i++) {
        cldnn::format fmt = static_cast<cldnn::format::type>(i);
        ASSERT_NO_THROW(fmt.to_string()) << "Can't convert to string format " << i;
    }
}

TEST(format, traits) {
    typedef std::underlying_type<cldnn::format::type>::type format_underlying_type;
    for (format_underlying_type i = 0; i < static_cast<format_underlying_type>(cldnn::format::format_num); i++) {
        cldnn::format fmt = static_cast<cldnn::format::type>(i);
        ASSERT_NO_THROW(cldnn::format::traits(fmt)) << "Can't get traits for format " << i;
    }
}

struct format_adjust_test_params {
    format in_format;
    size_t new_rank;
    format expected_format;
};

class format_adjust_test : public testing::TestWithParam<format_adjust_test_params> {
public:
    static std::string PrintToString(testing::TestParamInfo<format_adjust_test_params> param_info) {
        auto in_fmt = param_info.param.in_format.to_string();
        auto new_rank = std::to_string(param_info.param.new_rank);
        auto expected_fmt = param_info.param.expected_format.to_string();
        std::string res = "in_fmt=" + in_fmt + "_new_rank=" + new_rank + "_expected_fmt=" + expected_fmt;
        return res;
    }
 };

TEST_P(format_adjust_test, shape_infer) {
    auto p = GetParam();

    if (p.expected_format == format::any) {
        cldnn::format fmt = cldnn::format::any;
        ASSERT_ANY_THROW(fmt = format::adjust_to_rank(p.in_format, p.new_rank)) << fmt.to_string();
    } else {
        ASSERT_EQ(format::adjust_to_rank(p.in_format, p.new_rank), p.expected_format);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, format_adjust_test,
    testing::ValuesIn(std::vector<format_adjust_test_params>{
        {format::bfyx, 3, format::bfyx},
        {format::bfyx, 4, format::bfyx},
        {format::bfyx, 5, format::bfzyx},
        {format::bfyx, 6, format::bfwzyx},
        {format::bfzyx, 4, format::bfyx},
        {format::bfzyx, 6, format::bfwzyx},
        {format::bfwzyx, 3, format::bfyx},
        {format::b_fs_yx_fsv16, 5, format::b_fs_zyx_fsv16},
        {format::b_fs_zyx_fsv16, 4, format::b_fs_yx_fsv16},
        {format::fs_b_yx_fsv32, 5, format::any},
        {format::nv12, 5, format::any},
        {format::oiyx, 5, format::any},
    }),
    format_adjust_test::PrintToString);
