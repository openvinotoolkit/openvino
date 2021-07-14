// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>

#include "mkldnn_memory.h"
#include "cpu_memory_desc_utils.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

TEST(MemDescTest, Conversion) {
    // Check if conversion keep desc structure
    // dnnl::memory::desc -> MKLDNNMemoryDesc -> BlockedMemoryDesc -> MKLDNNMemoryDesc -> dnnl::memory::desc
    auto converted_correctly = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
        dnnl::memory::desc orig_tdesc {dims, dnnl::memory::data_type::u8, fmt};
        MKLDNNMemoryDesc plg_tdesc {orig_tdesc};
        BlockedMemoryDesc blk_tdesc = MemoryDescUtils::convertToBlockedDescriptor(plg_tdesc);
        MKLDNNMemoryDesc plg_tdesc_after = MemoryDescUtils::convertToMKLDNNMemoryDesc(blk_tdesc);
        dnnl::memory::desc after_tdesc(plg_tdesc_after);

        return  orig_tdesc == after_tdesc;
    };

    std::pair<dnnl::memory::format_tag, dnnl::memory::dims> payload[] {
        { dnnl::memory::format_tag::nChw16c,     {1, 1, 10, 10} },  // auto blocked
        { dnnl::memory::format_tag::nhwc,        {4, 2, 10, 7 } },  // permuted
        { dnnl::memory::format_tag::nchw,        {4, 2, 10, 7 } },  // plain
        { dnnl::memory::format_tag::NChw16n16c,  {4, 2, 10, 7 } },  // blocked for 2 dims
        { dnnl::memory::format_tag::BAcd16a16b,  {4, 2, 10, 7 } },  // blocked and permuted outer dims
        { dnnl::memory::format_tag::Acdb16a,     {96, 1, 7, 7 } },  // same strides but not default order
    };

    for (const auto &p : payload)
        ASSERT_TRUE(converted_correctly(p.first, p.second));
}

TEST(MemDescTest, CompareWithTensorDescRecomputedStrides) {
    auto converted_correctly = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
        dnnl::memory::desc orig_tdesc {dims, dnnl::memory::data_type::u8, fmt};
        MKLDNNMemoryDesc plg_tdesc {orig_tdesc};
        BlockedMemoryDesc blk_tdesc = MemoryDescUtils::convertToBlockedDescriptor(plg_tdesc);

        BlockedMemoryDesc recomputed_blk_tdesc(blk_tdesc.getPrecision(), blk_tdesc.getShape().getStaticDims(), blk_tdesc.getBlockDims(), blk_tdesc.getOrder());

        return  blk_tdesc.isCompatible(recomputed_blk_tdesc);
    };

    std::pair<dnnl::memory::format_tag, dnnl::memory::dims> payload[] {
        { dnnl::memory::format_tag::nChw16c,     {1, 1, 10, 10} },  // auto blocked
        { dnnl::memory::format_tag::nhwc,        {4, 2, 10, 7 } },  // permuted
        { dnnl::memory::format_tag::nchw,        {4, 2, 10, 7 } },  // plain
        { dnnl::memory::format_tag::NChw16n16c,  {4, 2, 10, 7 } },  // blocked for 2 dims
        { dnnl::memory::format_tag::BAcd16a16b,  {4, 2, 10, 7 } },  // blocked and permuted outer dims
        { dnnl::memory::format_tag::Acdb16a,     {96, 1, 7, 7 } },  // same strides but not default order
    };

    for (const auto &p : payload)
        ASSERT_TRUE(converted_correctly(p.first, p.second));
}

TEST(MemDescTest, isPlainCheck) {
    const auto dims = dnnl::memory::dims {3, 2, 5, 7};
    const auto type = dnnl::memory::data_type::u8;
    dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
    dnnl::memory::desc permt_tdesc {dims, type, dnnl::memory::format_tag::acdb};
    dnnl::memory::desc blckd_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};

    ASSERT_TRUE(MKLDNNMemoryDesc(plain_tdesc).checkGeneralLayout(GeneralLayout::ncsp));
    ASSERT_FALSE(MKLDNNMemoryDesc(permt_tdesc).checkGeneralLayout(GeneralLayout::ncsp));
    ASSERT_FALSE(MKLDNNMemoryDesc(blckd_tdesc).checkGeneralLayout(GeneralLayout::ncsp));
}

TEST(MemDescTest, isBlockedCCheck) {
    const auto dims = dnnl::memory::dims {3, 2, 5, 7};
    const auto type = dnnl::memory::data_type::u8;

    dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
    dnnl::memory::desc tailc_tdesc {dims, type, dnnl::memory::format_tag::acdb};
    dnnl::memory::desc blck8_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};
    dnnl::memory::desc blck8_permCD_tdesc {dims, type, dnnl::memory::format_tag::aBdc16b};
    const MKLDNNMemoryDesc plain_mdesc(plain_tdesc);
    const MKLDNNMemoryDesc tailc_mdesc(tailc_tdesc);
    ASSERT_FALSE(plain_mdesc.checkGeneralLayout(GeneralLayout::nCsp8c) || plain_mdesc.checkGeneralLayout(GeneralLayout::nCsp16c));
    ASSERT_FALSE(tailc_mdesc.checkGeneralLayout(GeneralLayout::nCsp8c) || tailc_mdesc.checkGeneralLayout(GeneralLayout::nCsp16c));
    ASSERT_TRUE(MKLDNNMemoryDesc(blck8_tdesc).checkGeneralLayout(GeneralLayout::nCsp8c));
    ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_tdesc).checkGeneralLayout(GeneralLayout::nCsp16c));

    const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
    const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};
    dnnl::memory::desc blck8_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
    dnnl::memory::desc blck8_permCD_crop_tdesc = blck8_permCD_tdesc.submemory_desc(crop_dims, crop_off);
    ASSERT_TRUE(MKLDNNMemoryDesc(blck8_crop_tdesc).checkGeneralLayout(GeneralLayout::nCsp8c));
    ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_crop_tdesc).checkGeneralLayout(GeneralLayout::nCsp8c));
}

TEST(MemDescTest, isTailCCheck) {
    const auto dims = dnnl::memory::dims {3, 2, 5, 7};
    const auto type = dnnl::memory::data_type::u8;

    dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
    dnnl::memory::desc tailc_tdesc {dims, type, dnnl::memory::format_tag::acdb};
    dnnl::memory::desc permt_tdesc {dims, type, dnnl::memory::format_tag::bcda};
    dnnl::memory::desc blck8_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};
    ASSERT_FALSE(MKLDNNMemoryDesc(plain_tdesc).checkGeneralLayout(GeneralLayout::nspc));
    ASSERT_FALSE(MKLDNNMemoryDesc(permt_tdesc).checkGeneralLayout(GeneralLayout::nspc));
    ASSERT_TRUE(MKLDNNMemoryDesc(tailc_tdesc).checkGeneralLayout(GeneralLayout::nspc));
    ASSERT_FALSE(MKLDNNMemoryDesc(blck8_tdesc).checkGeneralLayout(GeneralLayout::nspc));

    dnnl::memory::desc blck8_permCD_tdesc {dims, type, dnnl::memory::format_tag::aBdc16b};
    ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_tdesc).checkGeneralLayout(GeneralLayout::nspc));

    const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
    const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};
    dnnl::memory::desc tailc_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
    ASSERT_FALSE(MKLDNNMemoryDesc(tailc_crop_tdesc).checkGeneralLayout(GeneralLayout::nspc));
}

TEST(MemDescTest, constructWithPlainFormat) {
    GTEST_SKIP();
}

TEST(MemDescTest, CheckScalar) {
    GTEST_SKIP();
}

TEST(MemDescTest, UpperBound) {
    GTEST_SKIP();
}

TEST(MemDescTest, BlockedConversion) {
    GTEST_SKIP();
}

TEST(MemDescTest, ComaptibleWithFormat) {
    GTEST_SKIP();
}

TEST(isSameMethodTest, CheckTensorWithSameStrides) {
    auto isSameDataFormat = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
        dnnl::memory::desc oneDnnDesc {dims, dnnl::memory::data_type::u8, fmt};
        MKLDNNMemoryDesc pluginDesc {oneDnnDesc};
        return pluginDesc.getFormat() == fmt;
    };

    std::pair<dnnl::memory::format_tag, dnnl::memory::dims> testCases[] {
        { dnnl::memory::format_tag::ntc, {1, 10, 10} },
    };

    for (const auto &tc : testCases)
        ASSERT_TRUE(isSameDataFormat(tc.first, tc.second));
}
