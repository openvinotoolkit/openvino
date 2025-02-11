// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <oneapi/dnnl/dnnl.hpp>
#include <sstream>
#include <oneapi/dnnl/dnnl_debug.h>

#include "intel_gpu/runtime/utils.hpp"
#include "test_utils.h"

#include "intel_gpu/runtime/format.hpp"
#include "graph/impls/onednn/utils.hpp"

using namespace cldnn;

struct dnnl_desc_params {
    // Descriptor info to test
    dnnl::memory::dims dims;
    dnnl::memory::data_type data_type;
    dnnl::memory::dims strides;  // In case of plain (non-blocked) formats the strides between dimensions
};

struct desc_stride_test_params {
    dnnl_desc_params test_desc;
    std::vector<std::vector<size_t>> expected_orders;
};

class format_test_stride : public testing::TestWithParam<desc_stride_test_params> {
public:
    static std::string PrintToString(testing::TestParamInfo<desc_stride_test_params> param_info) {
        auto strides = param_info.param.test_desc.strides;
        std::string res = " stride : {";
        for (auto stride : strides) {
            res += std::to_string(stride) + " ";
        }
        res += "}";

        return res;
    }
};

TEST_P(format_test_stride, test_candidates_using_stride) {
    auto param = GetParam();

    dnnl::memory::desc test_desc(param.test_desc.dims, param.test_desc.data_type, param.test_desc.strides);
    auto candidates= onednn::get_candidate_orders(test_desc);

    ASSERT_EQ(candidates.size(), param.expected_orders.size());
    bool is_same = true;
    for (size_t idx = 0; idx < param.expected_orders.size(); idx++) {
        auto expected = param.expected_orders.at(idx);
        bool found_match = false;
        for (size_t idx = 0 ; idx < candidates.size() ; idx++) {
            if (std::equal(candidates[idx].begin(), candidates[idx].end(), expected.begin()))
                found_match = true;
        }

        if (!found_match)
            is_same = false;
    }

    ASSERT_TRUE(is_same);
}

INSTANTIATE_TEST_SUITE_P(smoke, format_test_stride,
    testing::ValuesIn(std::vector<desc_stride_test_params>{
        {{{1, 3, 8, 8}, dnnl::memory::data_type::f16, {768, 256, 16, 1}}, {{0, 1, 2, 3}}},
        {{{6, 1, 1, 8}, dnnl::memory::data_type::f16, {96, 16, 16, 1}}, {{0, 1, 2, 3}, {0, 2, 1, 3}}},
        {{{1, 1, 1, 16}, dnnl::memory::data_type::f16, {32, 32, 32, 1}}, {{0, 1, 2, 3}, {0, 2, 1, 3}, {1, 0, 2, 3}, {2, 1, 0, 3}, {1, 2, 0, 3}, {2, 0, 1, 3}}}
    }),
    format_test_stride::PrintToString);


struct format_matching_test_params {
    dnnl::memory::dims dims;
    dnnl::memory::data_type data_type;
    dnnl::memory::format_tag dnnl_format;
    cldnn::format cldnn_format;
};

class data_format_test_match_dnnl : public testing::TestWithParam<format_matching_test_params> {
public:
    static std::string PrintToString(testing::TestParamInfo<format_matching_test_params> param_info) {
        auto dnnl_format = param_info.param.dnnl_format;
        std::string res = " data format (dnnl::memory::format_tag) : " + std::string(dnnl_fmt_tag2str((dnnl_format_tag_t)dnnl_format));
        res += " > " + format::traits(param_info.param.cldnn_format).str;

        return res;
    }
};

TEST_P(data_format_test_match_dnnl, test_match_data_format) {
    auto param = GetParam();

    dnnl::memory::desc test_desc(param.dims, param.data_type, param.dnnl_format);
    auto result = onednn::find_data_format(test_desc);
    ASSERT_TRUE(result == param.cldnn_format);
}

INSTANTIATE_TEST_SUITE_P(smoke, data_format_test_match_dnnl,
    testing::ValuesIn(std::vector<format_matching_test_params>{
        {{1, 3, 8, 8}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::nchw, cldnn::format::bfyx},
        {{1, 3, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::aBcd4b, cldnn::format::b_fs_yx_fsv4},
        {{1, 12, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::nChw16c, cldnn::format::b_fs_yx_fsv16},
        {{1, 12, 16, 16}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::aBcd32b, cldnn::format::b_fs_yx_fsv32},
        {{1, 3, 16, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::aBcde4b, cldnn::format::b_fs_zyx_fsv4},
        {{12, 12, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::NChw16n16c, cldnn::format::bs_fs_yx_bsv16_fsv16},
        {{32, 32, 16, 16}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::NChw32n32c, cldnn::format::bs_fs_yx_bsv32_fsv32},
    }),
    data_format_test_match_dnnl::PrintToString);

class data_format_test_not_match_dnnl : public testing::TestWithParam<format_matching_test_params> {
public:
    static std::string PrintToString(testing::TestParamInfo<format_matching_test_params> param_info) {
        auto dnnl_format = param_info.param.dnnl_format;
        std::string res = " Failed case for " + std::string(dnnl_fmt_tag2str((dnnl_format_tag_t)dnnl_format));
        return res;
    }
};

TEST_P(data_format_test_not_match_dnnl, test_not_match_data_format) {
    auto param = GetParam();

    dnnl::memory::desc test_desc(param.dims, param.data_type, param.dnnl_format);
    auto result = onednn::find_data_format(test_desc);
    ASSERT_FALSE(result == param.cldnn_format);
}

INSTANTIATE_TEST_SUITE_P(smoke, data_format_test_not_match_dnnl,
    testing::ValuesIn(std::vector<format_matching_test_params>{
        {{1, 3, 8, 8}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::nchw, cldnn::format::byxf},
        {{1, 3, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::aBcd4b, cldnn::format::b_fs_yx_fsv2},
        {{1, 12, 16, 16}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::nChw16c, cldnn::format::b_fs_yx_fsv32},
        {{1, 12, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::aBcd32b, cldnn::format::b_fs_yx_fsv16},
        {{1, 3, 16, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::aBcde4b, cldnn::format::b_fs_zyx_fsv16},
        {{32, 32, 16, 16}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::NChw16n16c, cldnn::format::bs_fs_yx_bsv32_fsv32},
        {{12, 12, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::NChw32n32c, cldnn::format::bs_fs_yx_bsv16_fsv16},
    }),
    data_format_test_not_match_dnnl::PrintToString);

class weight_format_test_match_dnnl : public testing::TestWithParam<format_matching_test_params> {
public:
    static std::string PrintToString(testing::TestParamInfo<format_matching_test_params> param_info) {
        auto dnnl_format = param_info.param.dnnl_format;
        std::string res = " weight format (dnnl::memory::format_tag) : " + std::string(dnnl_fmt_tag2str((dnnl_format_tag_t)dnnl_format));
        res += " > " + format::traits(param_info.param.cldnn_format).str;

        return res;
    }
};

TEST_P(weight_format_test_match_dnnl, test_match_data_format) {
    auto param = GetParam();

    dnnl::memory::desc test_desc(param.dims, param.data_type, param.dnnl_format);
    auto result = onednn::find_format(test_desc, false);
    ASSERT_TRUE(result == param.cldnn_format);
}

INSTANTIATE_TEST_SUITE_P(smoke, weight_format_test_match_dnnl,
    testing::ValuesIn(std::vector<format_matching_test_params>{
        {{1, 3, 8, 8}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::abcd, cldnn::format::oiyx},
        {{16, 16, 8, 8}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::ABcd16b16a, cldnn::format::os_is_yx_isv16_osv16},
    }),
    weight_format_test_match_dnnl::PrintToString);

struct memory_desc_to_fmt_conversion_params {
    dnnl::memory::dims dims;
    dnnl::memory::data_type data_type;
    dnnl::memory::format_tag dnnl_format;
    bool grouped;
    cldnn::format_traits expected;
};

class memory_desc_to_fmt_conversion_test : public testing::TestWithParam<memory_desc_to_fmt_conversion_params> {
public:
    static std::string PrintToString(testing::TestParamInfo<memory_desc_to_fmt_conversion_params> param_info) {
        auto dnnl_format = param_info.param.dnnl_format;
        std::stringstream s;
        ov::Shape shape(param_info.param.dims.begin(), param_info.param.dims.end());
        s << shape;
        return std::string(dnnl_fmt_tag2str((dnnl_format_tag_t)dnnl_format)) + "_" + s.str();
    }
};

TEST_P(memory_desc_to_fmt_conversion_test, test_match_data_format) {
    auto param = GetParam();

    dnnl::memory::desc test_desc(param.dims, param.data_type, param.dnnl_format);
    auto result = onednn::convert_memory_desc_to_traits(test_desc, true, param.grouped);
    ASSERT_EQ(result.str, param.expected.str);
    ASSERT_EQ(result.batch_num, param.expected.batch_num);
    ASSERT_EQ(result.feature_num, param.expected.feature_num);
    ASSERT_EQ(result.spatial_num, param.expected.spatial_num);
    ASSERT_EQ(result.group_num, param.expected.group_num);
    ASSERT_EQ(result._order, param.expected._order);
    ASSERT_EQ(result.order, param.expected.order);
    ASSERT_EQ(result.internal_order, param.expected.internal_order);
    ASSERT_EQ(result.block_sizes, param.expected.block_sizes);
    ASSERT_EQ(result.logic_block_sizes, param.expected.logic_block_sizes);
}

INSTANTIATE_TEST_SUITE_P(smoke, memory_desc_to_fmt_conversion_test,
    testing::ValuesIn(std::vector<memory_desc_to_fmt_conversion_params>{
        {
            {1, 3, 8, 8}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::abcd, false,
            format_traits{
                "custom", 1, 1, 2, 0, {0, 1, 2, 3}, "oiyx", "oixy?", {}, {}
            }
        },
        {
            {16, 16, 8, 8}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::ABcd16b16a, false,
            format_traits{
                "custom", 1, 1, 2, 0, {0, 1, 2, 3}, "oiyx", "oixy?", {{1, 16}, {0, 16}}, {{1, 16}, {0, 16}}
            }
        },
        {
            {8, 4, 16, 16, 16}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ABcde8a4b, false,
            format_traits{
                "custom", 1, 1, 3, 0, {0, 1, 2, 3, 4}, "oizyx", "oixyz", {{0, 8}, {1, 4}}, {{0, 8}, {1, 4}}
            }
        },
        {
            {16, 16, 8, 8}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ABcd2a8b8a2b, false,
            format_traits{
                "custom", 1, 1, 2, 0, {0, 1, 2, 3}, "oiyx", "oixy?", {{0, 2}, {1, 8}, {0, 8}, {1, 2}}, {{0, 2}, {1, 8}, {0, 8}, {1, 2}}
            }
        },
        {
            {64, 64, 8, 8}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::BAcd4b8a8b4a, false,
            format_traits{
                "custom", 1, 1, 2, 0, {1, 0, 2, 3}, "ioyx", "oixy?", {{1, 4}, {0, 8}, {1, 8}, {0, 4}}, {{1, 4}, {0, 8}, {1, 8}, {0, 4}}
            }
        },
        {
            {32, 32, 8, 8}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::BAcd4b8a8b4a, false,  // same format as above but different sizes
            format_traits{
                "custom", 1, 1, 2, 0, {0, 1, 2, 3}, "oiyx", "oixy?", {{1, 4}, {0, 8}, {1, 8}, {0, 4}}, {{1, 4}, {0, 8}, {1, 8}, {0, 4}} // Order is different due to A=B=1 (strides are equal)
            }
        },
        {
            {65, 32, 8, 8}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::BAcd4b8a8b4a, false,  // same format as above but different sizes
            format_traits{
                "custom", 1, 1, 2, 0, {1, 0, 2, 3}, "ioyx", "oixy?", {{1, 4}, {0, 8}, {1, 8}, {0, 4}}, {{1, 4}, {0, 8}, {1, 8}, {0, 4}}
            }
        },
        {
            {10, 20, 30, 40, 50}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::AcdeB24a2b, true,
            format_traits{
                "custom", 1, 1, 2, 1, {0, 2, 3, 4, 1}, "giyxo", "oixy????g", {{8, 24}, {0, 2}}, {{0, 24}, {1, 2}}
            }
        },
        {
            {10, 60, 70, 40, 50}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::aCBde4c8b8c4b, true,
            format_traits{
                "custom", 1, 1, 2, 1, {0, 2, 1, 3, 4}, "gioyx", "oixy????g", {{1, 4}, {0, 8}, {1, 8}, {0, 4}}, {{2, 4}, {1, 8}, {2, 8}, {1, 4}}
            }
        },
    }),
    memory_desc_to_fmt_conversion_test::PrintToString);
