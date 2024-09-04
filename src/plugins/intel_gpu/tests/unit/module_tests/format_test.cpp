// Copyright (C) 2018-2024 Intel Corporation
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
    cldnn::format in_format;
    size_t new_rank;
    cldnn::format expected_format;
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

    if (p.expected_format == cldnn::format::any) {
        cldnn::format fmt = cldnn::format::any;
        ASSERT_ANY_THROW(fmt = format::adjust_to_rank(p.in_format, p.new_rank)) << fmt.to_string();
    } else {
        ASSERT_EQ(format::adjust_to_rank(p.in_format, p.new_rank), p.expected_format);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, format_adjust_test,
    testing::ValuesIn(std::vector<format_adjust_test_params>{
        {cldnn::format::bfyx,           3, cldnn::format::bfyx},
        {cldnn::format::bfyx,           4, cldnn::format::bfyx},
        {cldnn::format::bfyx,           5, cldnn::format::bfzyx},
        {cldnn::format::bfyx,           6, cldnn::format::bfwzyx},
        {cldnn::format::bfzyx,          4, cldnn::format::bfyx},
        {cldnn::format::bfzyx,          6, cldnn::format::bfwzyx},
        {cldnn::format::bfwzyx,         3, cldnn::format::bfyx},
        {cldnn::format::b_fs_yx_fsv16,  5, cldnn::format::b_fs_zyx_fsv16},
        {cldnn::format::b_fs_zyx_fsv16, 4, cldnn::format::b_fs_yx_fsv16},
        {cldnn::format::fs_b_yx_fsv32,  5, cldnn::format::any},
        {cldnn::format::nv12,           5, cldnn::format::any},
        {cldnn::format::oiyx,           5, cldnn::format::any},
    }),
    format_adjust_test::PrintToString);

struct axes_test_format_params {
    cldnn::format in_format;
    // First : block_idx, Second : block_size
    std::vector<std::pair<size_t, int>> inner_block;
    std::vector<std::pair<size_t, int>> per_axis_block;
};

class axes_test_format : public testing::TestWithParam<axes_test_format_params> {
public:
    static std::string PrintToString(testing::TestParamInfo<axes_test_format_params> param_info) {
        auto input_fmt = param_info.param.in_format.to_string();
        auto blocks = param_info.param.inner_block;
        auto per_axis_blocks = param_info.param.per_axis_block;
        std::string res = "in_fmt = " + input_fmt + " : ";
        for (auto block : blocks) {
            res += " { " + std::to_string(block.first) + ", " + std::to_string(block.second) + "}";
        }

        res += " > ";
        for (auto block : per_axis_blocks) {
            res += " { " + std::to_string(block.first) + ", " + std::to_string(block.second) + "}";
        }

        return res;
    }
 };

 TEST_P(axes_test_format, simple_test) {
    auto param = GetParam();

    auto per_axis_blocks = format::per_axis_block_size(param.in_format);
    ASSERT_EQ(per_axis_blocks.size(), param.per_axis_block.size());
    for (size_t idx = 0; idx < per_axis_blocks.size(); idx++) {
        ASSERT_EQ(per_axis_blocks.at(idx).first, param.per_axis_block.at(idx).first);
        ASSERT_EQ(per_axis_blocks.at(idx).second, param.per_axis_block.at(idx).second);
    }

    auto blocks = format::block_sizes(param.in_format);
    ASSERT_EQ(blocks.size(), param.inner_block.size());
    for (size_t idx = 0; idx < blocks.size(); idx++) {
        ASSERT_EQ(blocks.at(idx).first, param.inner_block.at(idx).first);
        ASSERT_EQ(blocks.at(idx).second, param.inner_block.at(idx).second);
    }

    auto logic_blocks = format::logic_block_sizes(param.in_format);
    ASSERT_EQ(logic_blocks.size(), param.inner_block.size());
    for (size_t idx = 0; idx < logic_blocks.size(); idx++) {
        auto c = param.in_format.internal_order()[param.inner_block.at(idx).first];
        auto pos = param.in_format.order().find(c);
        if (pos == std::string::npos)
            throw std::domain_error(std::string("Unknown coord type: ") + c);

        auto expected_logic_idx = param.in_format.dims_order()[pos];
        auto expected_logic_size = param.inner_block.at(idx).second;

        ASSERT_EQ(logic_blocks.at(idx).first, expected_logic_idx);
        ASSERT_EQ(logic_blocks.at(idx).second, expected_logic_size);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, axes_test_format,
    testing::ValuesIn(std::vector<axes_test_format_params>{
        {format::os_is_yx_isa8_osv8_isv4,       {{1, 8}, {0, 8}, {1, 4}}, {{1, 32}, {0, 8}}},
        {format::os_is_yx_isa8_osv16_isv4,      {{1, 8}, {0, 16}, {1, 4}}, {{1, 32}, {0, 16}}},
        {format::os_is_yx_osa4_isa8_osv8_isv4,  {{0, 4}, {1, 8}, {0, 8}, {1, 4}}, {{0, 32}, {1, 32}}},
        {format::os_is_zyx_osa4_isa8_osv8_isv4, {{0, 4}, {1, 8}, {0, 8}, {1, 4}}, {{0, 32}, {1, 32}}},
        {format::os_is_zyx_isa8_osv8_isv4,      {{1, 8}, {0, 8}, {1, 4}}, {{1, 32}, {0, 8}}},
        {format::os_is_zyx_isa8_osv16_isv4,     {{1, 8}, {0, 16}, {1, 4}}, {{1, 32}, {0, 16}}},
        {format::os_is_yx_osv8_isv4,            {{0, 8}, {1, 4}}, {{0, 8}, {1, 4}}},
        {format::gs_oiyx_gsv32,                   {{8, 32}}, {{8, 32}}},
        {format::gs_oiyx_gsv16,                   {{8, 16}}, {{8, 16}}},
        {format::gs_oizyx_gsv16,                  {{8, 16}}, {{8, 16}}},
        {format::i_yxs_os_yxsv2_osv16,            {{0, 16}}, {{0, 16}}},
        {format::iy_xs_os_xsv2_osv8__ao32,        {{2, 2}, {0, 8}},   {{2, 2}, {0, 8}}},
        {format::g_is_os_zyx_isv16_osv16,         {{1, 16}, {0, 16}}, {{1, 16}, {0, 16}}},
        {format::g_is_os_yx_isv16_osv16,          {{1, 16}, {0, 16}}, {{1, 16}, {0, 16}}},
        {format::gi_yxs_os_yxsv2_osv16,           {{0, 16}},          {{0, 16}}},
        {format::giy_xs_os_xsv2_osv8__ao32,       {{2, 2}, {0, 8}},   {{2, 2}, {0, 8}}},
    }),
    axes_test_format::PrintToString);

struct find_format_test_params {
    std::vector<uint64_t> dims_order;
    std::vector<std::pair<size_t, int>> block_sizes;
    bool is_weights;
    bool is_grouped;
    bool is_image_2d;
    bool is_winograd;
    bool is_nv12;
    format expected_format;
};

class find_format_test : public testing::TestWithParam<find_format_test_params> {
public:
    static std::string PrintToString(testing::TestParamInfo<find_format_test_params> param_info) {
        auto& p = param_info.param;

        std::string res = "order = [";
        for (size_t i = 0; i < p.dims_order.size() - 1; ++i) {
            res += std::to_string(p.dims_order[i]) + ", ";
        }
        res += std::to_string(p.dims_order.back()) + "], ";

        res += "block_sizes = [";
        for (auto block : p.block_sizes) {
            res += "{ " + std::to_string(block.first) + ", " + std::to_string(block.second) + "}";
        }
        res += "], ";

        res += "is_weights = " + std::to_string(p.is_weights) + ", " +
               "is_grouped = " + std::to_string(p.is_grouped) + ", " +
               "is_image_2d = " + std::to_string(p.is_image_2d) + ", " +
               "is_winograd = " + std::to_string(p.is_winograd) + ", " +
               "is_nv12 = " + std::to_string(p.is_nv12) + ", " +
               "expected_format = " + param_info.param.expected_format.to_string();

        return res;
    }
 };

TEST_P(find_format_test, simple_test) {
    auto p = GetParam();

    if (p.expected_format == format::any) {
        ASSERT_ANY_THROW(format::find_format(p.dims_order, p.block_sizes,
                                             p.is_weights, p.is_grouped, p.is_image_2d, p.is_winograd, p.is_nv12));
    } else {
        ASSERT_EQ(format::find_format(p.dims_order, p.block_sizes,
                                      p.is_weights, p.is_grouped, p.is_image_2d, p.is_winograd, p.is_nv12), p.expected_format);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, find_format_test,
    testing::ValuesIn(std::vector<find_format_test_params>{
    //   Dims order             Block sizes               is_weights is_grouped is_image_2d is_winograd is_nv12   Expected format
        {{0, 1, 2, 3},          {},                       false,     false,     false,      false,      false,    format::bfyx},
        {{1, 0, 2, 3},          {},                       false,     false,     false,      false,      false,    format::fbyx},
        {{2, 3, 1, 0},          {},                       false,     false,     false,      false,      false,    format::yxfb},
        {{0, 2, 3, 1},          {},                       false,     false,     false,      false,      false,    format::byxf},
        {{1, 2, 3, 0},          {},                       false,     false,     false,      false,      false,    format::fyxb},
        {{0, 2, 1, 3},          {},                       false,     false,     false,      false,      false,    format::byfx},
        {{0, 3, 1, 2},          {},                       false,     false,     false,      false,      false,    format::bxfy},
        {{0, 1, 2, 3},          {{1, 16}},                false,     false,     false,      false,      false,    format::b_fs_yx_fsv16},
        {{0, 1},                {{0, 8}, {1, 8}},         false,     false,     false,      false,      false,    format::bs_fs_fsv8_bsv8},
        {{0, 1, 2, 3, 4, 5, 6}, {},                       false,     false,     false,      false,      false,    format::bfuwzyx},
        {{0, 1, 2, 3},          {},                       false,     false,     true,       false,      true,     format::nv12},
        {{0, 1, 2, 3},          {},                       false,     false,     true,       false,      false,    format::image_2d_rgba},
        {{0, 1, 2, 3},          {},                       true,      false,     false,      false,      false,    format::oiyx},
        {{1, 0, 2, 3},          {},                       true,      false,     false,      false,      false,    format::ioyx},
        {{1, 2, 3, 0},          {},                       true,      false,     false,      false,      false,    format::iyxo},
        {{0, 2, 3, 1},          {},                       true,      false,     false,      false,      false,    format::oyxi},
        {{0, 2, 1, 3},          {},                       true,      false,     false,      false,      false,    format::oyix},
        {{0, 3, 1, 2},          {},                       true,      false,     false,      false,      false,    format::oxiy},
        {{2, 3, 1, 0},          {},                       true,      false,     false,      false,      false,    format::yxio},
        {{0, 1, 2, 3},          {{0, 16}},                true,      false,     false,      false,      false,    format::os_iyx_osv16},
        {{0, 1, 2, 3},          {},                       true,      false,     false,      true,       false,    format::winograd_2x3_s1_weights},
        {{0, 1, 2, 3},          {{1, 8}, {0, 8}, {1, 4}}, true,      false,     false,      false,      false,    format::os_is_yx_isa8_osv8_isv4},
        {{0, 1, 2, 3, 4},       {},                       true,      true,      false,      false,      false,    format::goiyx},
        {{0, 2, 1, 3, 4},       {{1, 16}, {0, 16}},       true,      true,      false,      false,      false,    format::g_is_os_yx_isv16_osv16},

    //  Expected error throw
        {{0, 0, 2, 3},          {},                       false,     false,     false,     false,       false,    format::any},
        {{0, 1, 2, 3},          {{1, 1}},                 false,     false,     false,     false,       false,    format::any},
        {{0, 1, 2, 3},          {},                       false,     true,      false,     false,       false,    format::any}
    }),
    find_format_test::PrintToString);
