// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "mkldnn_memory.h"
#include "cpu_memory_desc_utils.h"
#include "nodes/common/blocked_desc_creator.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace testing;

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

        BlockedMemoryDesc recomputed_blk_tdesc(blk_tdesc.getPrecision(), blk_tdesc.getShape(), blk_tdesc.getBlockDims(), blk_tdesc.getOrder());

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

    ASSERT_TRUE(MKLDNNMemoryDesc(plain_tdesc).hasLayoutType(LayoutType::ncsp));
    ASSERT_FALSE(MKLDNNMemoryDesc(permt_tdesc).hasLayoutType(LayoutType::ncsp));
    ASSERT_FALSE(MKLDNNMemoryDesc(blckd_tdesc).hasLayoutType(LayoutType::ncsp));
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
    ASSERT_FALSE(plain_mdesc.hasLayoutType(LayoutType::nCsp8c) || plain_mdesc.hasLayoutType(LayoutType::nCsp16c));
    ASSERT_FALSE(tailc_mdesc.hasLayoutType(LayoutType::nCsp8c) || tailc_mdesc.hasLayoutType(LayoutType::nCsp16c));
    ASSERT_TRUE(MKLDNNMemoryDesc(blck8_tdesc).hasLayoutType(LayoutType::nCsp8c));
    ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_tdesc).hasLayoutType(LayoutType::nCsp16c));

    const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
    const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};
    dnnl::memory::desc blck8_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
    dnnl::memory::desc blck8_permCD_crop_tdesc = blck8_permCD_tdesc.submemory_desc(crop_dims, crop_off);
    ASSERT_TRUE(MKLDNNMemoryDesc(blck8_crop_tdesc).hasLayoutType(LayoutType::nCsp8c));
    ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_crop_tdesc).hasLayoutType(LayoutType::nCsp8c));
}

TEST(MemDescTest, isTailCCheck) {
    const auto dims = dnnl::memory::dims {3, 2, 5, 7};
    const auto type = dnnl::memory::data_type::u8;

    dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
    dnnl::memory::desc tailc_tdesc {dims, type, dnnl::memory::format_tag::acdb};
    dnnl::memory::desc permt_tdesc {dims, type, dnnl::memory::format_tag::bcda};
    dnnl::memory::desc blck8_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};
    ASSERT_FALSE(MKLDNNMemoryDesc(plain_tdesc).hasLayoutType(LayoutType::nspc));
    ASSERT_FALSE(MKLDNNMemoryDesc(permt_tdesc).hasLayoutType(LayoutType::nspc));
    ASSERT_TRUE(MKLDNNMemoryDesc(tailc_tdesc).hasLayoutType(LayoutType::nspc));
    ASSERT_FALSE(MKLDNNMemoryDesc(blck8_tdesc).hasLayoutType(LayoutType::nspc));

    dnnl::memory::desc blck8_permCD_tdesc {dims, type, dnnl::memory::format_tag::aBdc16b};
    ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_tdesc).hasLayoutType(LayoutType::nspc));

    const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
    const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};
    dnnl::memory::desc tailc_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
    ASSERT_FALSE(MKLDNNMemoryDesc(tailc_crop_tdesc).hasLayoutType(LayoutType::nspc));
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

TEST(MKLDNNMemDescTest, KeepOrder) {
    using mkldnn::memory;
    std::vector<size_t> dims = {7, 3, 1, 5};
    memory::data_type dataType = memory::data_type::u8;
    MKLDNNMemoryDesc descPalanar(dims, dataType);
    ASSERT_THAT(descPalanar.getOrder(), ElementsAre(0, 1, 2, 3));

    MKLDNNMemoryDesc descTailC(dims, dataType, memory::format_tag::acdb);
    ASSERT_THAT(descTailC.getOrder(), ElementsAre(0, 2, 3, 1));

    MKLDNNMemoryDesc descBlockedC(dims, dataType, memory::format_tag::aBcd16b);
    ASSERT_THAT(descBlockedC.getOrder(), ElementsAre(0, 1, 2, 3, 1));

    MKLDNNMemoryDesc descWeightBlocked(dims, dataType, memory::format_tag::ABcd16b16a2b);
    ASSERT_THAT(descWeightBlocked.getOrder(), ElementsAre(0, 1, 2, 3, 1, 0, 1));

    auto dnnDims = MKLDNNExtensionUtils::convertToDnnlDims(dims);

    memory::desc mkldnnDescPlanar(dnnDims, dataType, memory::format_tag::abcd);
    ASSERT_THAT(MKLDNNMemoryDesc(mkldnnDescPlanar).getOrder(), ElementsAre(0, 1, 2, 3));

    memory::desc mkldnnDescTailC(dnnDims, dataType, memory::format_tag::acdb);
    ASSERT_THAT(MKLDNNMemoryDesc(mkldnnDescTailC).getOrder(), ElementsAre(0, 2, 3, 1));

    memory::desc mkldnnDescBlockedC(dnnDims, dataType, memory::format_tag::aBcd16b);
    ASSERT_THAT(MKLDNNMemoryDesc(mkldnnDescBlockedC).getOrder(), ElementsAre(0, 1, 2, 3, 1));

    memory::desc mkldnnDescWeightBlocked(dnnDims, dataType, memory::format_tag::ABcd16b16a2b);
    ASSERT_THAT(MKLDNNMemoryDesc(mkldnnDescWeightBlocked).getOrder(), ElementsAre(0, 1, 2, 3, 1, 0, 1));
}

TEST(MemDescTest, UndefinedState) {
    ngraph::PartialShape ngraphShape({{16}, {-1, -1}, {20, 30}, {7}});
    MKLDNNPlugin::Shape pluginShape(ngraphShape);
    MKLDNNMemoryDesc memDesc(pluginShape, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::nChw8c);

    ASSERT_FALSE(memDesc.isDefined());

    ASSERT_THROW(memDesc.cloneWithNewDims({16, 7, 40, 7}), InferenceEngine::ParameterMismatch);
    ASSERT_THROW(memDesc.cloneWithNewDims({16, 7, 25}), InferenceEngine::ParameterMismatch);
    ASSERT_THROW(memDesc.cloneWithNewDims({16, 7, 25, 5}), InferenceEngine::ParameterMismatch);

    auto definedDesc = memDesc.cloneWithNewDims({16, 15, 25, 7});

    ASSERT_TRUE(definedDesc->isDefined());

    auto creator = BlockedDescCreator::getCommonCreators().at(LayoutType::nCsp8c);
    auto blockedDesc = creator->createDesc(Precision::FP32, pluginShape);

    ASSERT_FALSE(blockedDesc.isDefined());

    ASSERT_TRUE(blockedDesc.isCompatible(memDesc));

    ASSERT_THROW(blockedDesc.cloneWithNewDims({16, 7, 40, 7}), InferenceEngine::ParameterMismatch);
    ASSERT_THROW(blockedDesc.cloneWithNewDims({16, 7, 25}), InferenceEngine::ParameterMismatch);
    ASSERT_THROW(blockedDesc.cloneWithNewDims({16, 7, 25, 5}), InferenceEngine::ParameterMismatch);

    auto definedBlockedDesc = blockedDesc.cloneWithNewDims({16, 15, 25, 7});

    ASSERT_TRUE(definedBlockedDesc->isDefined());

    ASSERT_FALSE(memDesc.isCompatible(*definedDesc));
    ASSERT_FALSE(memDesc.isCompatible(*definedBlockedDesc));

    ASSERT_TRUE(definedBlockedDesc->isCompatible(*definedDesc));
}

TEST(MemDescTest, MemSize) {
    constexpr size_t undefSize = MemoryDesc::UNDEFINED_SIZE;
    static const auto dnnlDataType = mkldnn::memory::data_type::f32;
    static const Precision iePrc = Precision::FP32;


    ngraph::PartialShape ngraphShapeUndef({{16}, {-1, -1}, {20, 30}, {7}});
    MKLDNNPlugin::Shape pluginShapeUndef(ngraphShapeUndef);

    auto creator = BlockedDescCreator::getCommonCreators().at(LayoutType::nspc);
    auto blockedDescUndef = creator->createDesc(iePrc, pluginShapeUndef);

    ASSERT_EQ(blockedDescUndef.getCurrentSize(), undefSize);
    ASSERT_EQ(blockedDescUndef.getMaxMemSize(), undefSize);

    MKLDNNMemoryDesc memDescUndef(pluginShapeUndef, dnnlDataType, mkldnn::memory::format_tag::nhwc);

    ASSERT_EQ(memDescUndef.getCurrentSize(), undefSize);
    ASSERT_EQ(memDescUndef.getMaxMemSize(), undefSize);

    ngraph::PartialShape ngraphShapeDefUpperBound({{16}, {7, 14}, {20, 30}, {7}});
    MKLDNNPlugin::Shape pluginShapeDefUpperBound(ngraphShapeDefUpperBound);

    auto blockedDescDefUpper = creator->createDesc(iePrc, pluginShapeDefUpperBound);

    ASSERT_EQ(blockedDescDefUpper.getCurrentSize(), undefSize);
    auto maxElementsCount = std::accumulate(pluginShapeDefUpperBound.getMaxDims().begin(),
                                            pluginShapeDefUpperBound.getMaxDims().end(),
                                            1, std::multiplies<size_t>());
    ASSERT_EQ(blockedDescDefUpper.getMaxMemSize(), maxElementsCount * iePrc.size());

    MKLDNNMemoryDesc memDescDefUpper(pluginShapeDefUpperBound, dnnlDataType, mkldnn::memory::format_tag::nhwc);

    ASSERT_EQ(memDescDefUpper.getCurrentSize(), undefSize);
    ASSERT_EQ(memDescDefUpper.getMaxMemSize(), maxElementsCount * MKLDNNExtensionUtils::sizeOfDataType(dnnlDataType));

    ngraph::PartialShape ngraphShapeDefined({{16}, {16}, {10}, {7}});
    MKLDNNPlugin::Shape pluginShapeDefined(ngraphShapeDefined);

    auto blockedDescDefined = creator->createDesc(iePrc, pluginShapeDefined);

    ASSERT_NE(blockedDescDefined.getCurrentSize(), undefSize);
    ASSERT_NE(blockedDescDefined.getMaxMemSize(), undefSize);
    ASSERT_EQ(blockedDescDefined.getCurrentSize(), blockedDescDefined.getMaxMemSize());

    MKLDNNMemoryDesc memDescDefined(pluginShapeDefined, dnnlDataType, mkldnn::memory::format_tag::nhwc);

    ASSERT_NE(memDescDefined.getCurrentSize(), undefSize);
    ASSERT_NE(memDescDefined.getMaxMemSize(), undefSize);
    ASSERT_EQ(memDescDefined.getCurrentSize(), memDescDefined.getMaxMemSize());
    ASSERT_EQ(blockedDescDefined.getCurrentSize(), memDescDefined.getCurrentSize());
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
