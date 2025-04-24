// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include <cpu_memory.h>
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/blocked_desc_creator.h"
#include <dnnl_extension_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "common_test_utils/test_assertions.hpp"

using namespace ov::intel_cpu;
using namespace testing;

TEST(MemDescTest, Conversion) {
    // Check if conversion keep desc structure
    // dnnl::memory::desc -> DnnlBlockedMemoryDesc -> CpuBlockedMemoryDesc -> DnnlBlockedMemoryDesc -> dnnl::memory::desc
    auto converted_correctly = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
        dnnl::memory::desc orig_tdesc {dims, dnnl::memory::data_type::u8, fmt};
        DnnlMemoryDescPtr plg_tdesc = DnnlExtensionUtils::makeDescriptor(orig_tdesc);
        BlockedMemoryDescPtr blk_tdesc = MemoryDescUtils::convertToBlockedMemoryDesc(plg_tdesc);
        MemoryDescPtr cpu_blk_tdesc = std::make_shared<CpuBlockedMemoryDesc>(blk_tdesc->getPrecision(), blk_tdesc->getShape(), blk_tdesc->getBlockDims(),
                                                                  blk_tdesc->getOrder(), blk_tdesc->getOffsetPadding(), blk_tdesc->getOffsetPaddingToData(),
                                                                  blk_tdesc->getStrides());
        DnnlMemoryDescPtr plg_tdesc_after = MemoryDescUtils::convertToDnnlMemoryDesc(cpu_blk_tdesc);
        dnnl::memory::desc after_tdesc = plg_tdesc_after->getDnnlDesc();

        return orig_tdesc == after_tdesc;
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

TEST(MemDescTest, UndefinedStateConversion) {
    ov::PartialShape ngraphUndefinedShape({{16}, {7, 15}, {-1, -1}, {3}});
    Shape cpuShape(ngraphUndefinedShape);

    const std::vector<dnnl::memory::format_tag> vecTags = {
            dnnl::memory::format_tag::nChw8c,
            dnnl::memory::format_tag::nhwc,
            dnnl::memory::format_tag::nChw16c,
            dnnl::memory::format_tag::ABcd16a16b,
            dnnl::memory::format_tag::OIhw4i16o4i
    };

    for (auto tag : vecTags) {
        DnnlBlockedMemoryDescPtr dnnlBlockedDesc = std::make_shared<DnnlBlockedMemoryDesc>(cpuShape, dnnl::memory::data_type::f32, tag);

        ASSERT_FALSE(dnnlBlockedDesc->isDefined());

        auto blockedDesc = MemoryDescUtils::convertToBlockedMemoryDesc(dnnlBlockedDesc);
        MemoryDescPtr cpuBlockedDesc = std::make_shared<CpuBlockedMemoryDesc>(blockedDesc->getPrecision(), blockedDesc->getShape(), blockedDesc->getBlockDims(),
                                                                    blockedDesc->getOrder(), blockedDesc->getOffsetPadding(),
                                                                    blockedDesc->getOffsetPaddingToData(), blockedDesc->getStrides());

        ASSERT_TRUE(dnnlBlockedDesc->isCompatible(*cpuBlockedDesc));
        ASSERT_TRUE(cpuBlockedDesc->isCompatible(*dnnlBlockedDesc));

        auto reconstructedDesc = MemoryDescUtils::convertToDnnlMemoryDesc(cpuBlockedDesc);

        ASSERT_TRUE(dnnlBlockedDesc->isCompatible(*reconstructedDesc));
        ASSERT_TRUE(cpuBlockedDesc->isCompatible(*reconstructedDesc));

        dnnl::memory::desc dnnlDesc = dnnlBlockedDesc->getDnnlDesc();
        dnnl::memory::desc reconstDnnlDesc = reconstructedDesc->getDnnlDesc();

        ASSERT_EQ(dnnlDesc, reconstDnnlDesc);

        auto definedMemDesc = dnnlBlockedDesc->cloneWithNewDims({16, 10, 15, 3});
        auto definedReconstructedDnnlDesc = reconstructedDesc->cloneWithNewDims({16, 10, 15, 3});

        ASSERT_TRUE(definedMemDesc->isCompatible(*definedReconstructedDnnlDesc));
    }
}

TEST(MemDescTest, CompareWithTensorDescRecomputedStrides) {
    auto converted_correctly = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
        dnnl::memory::desc orig_tdesc {dims, dnnl::memory::data_type::u8, fmt};
        DnnlMemoryDescPtr plg_tdesc = DnnlExtensionUtils::makeDescriptor(orig_tdesc);
        BlockedMemoryDescPtr blk_tdesc = MemoryDescUtils::convertToBlockedMemoryDesc(plg_tdesc);

        CpuBlockedMemoryDesc recomputed_blk_tdesc(blk_tdesc->getPrecision(), blk_tdesc->getShape(), blk_tdesc->getBlockDims(), blk_tdesc->getOrder());

        return  plg_tdesc->isCompatible(recomputed_blk_tdesc);
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

    ASSERT_TRUE(DnnlExtensionUtils::makeDescriptor(plain_tdesc)->hasLayoutType(LayoutType::ncsp));
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(permt_tdesc)->hasLayoutType(LayoutType::ncsp));
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(blckd_tdesc)->hasLayoutType(LayoutType::ncsp));
}

TEST(MemDescTest, isBlockedCCheck) {
    const auto dims = dnnl::memory::dims {3, 2, 5, 7};
    const auto type = dnnl::memory::data_type::u8;

    dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
    dnnl::memory::desc tailc_tdesc {dims, type, dnnl::memory::format_tag::acdb};
    dnnl::memory::desc blck8_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};
    dnnl::memory::desc blck8_permCD_tdesc {dims, type, dnnl::memory::format_tag::aBdc16b};
    auto plain_mdesc = DnnlExtensionUtils::makeDescriptor(plain_tdesc);
    auto tailc_mdesc = DnnlExtensionUtils::makeDescriptor(tailc_tdesc);
    ASSERT_FALSE(plain_mdesc->hasLayoutType(LayoutType::nCsp8c) || plain_mdesc->hasLayoutType(LayoutType::nCsp16c));
    ASSERT_FALSE(tailc_mdesc->hasLayoutType(LayoutType::nCsp8c) || tailc_mdesc->hasLayoutType(LayoutType::nCsp16c));
    ASSERT_TRUE(DnnlExtensionUtils::makeDescriptor(blck8_tdesc)->hasLayoutType(LayoutType::nCsp8c));
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(blck8_permCD_tdesc)->hasLayoutType(LayoutType::nCsp16c));

    const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
    const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};
    dnnl::memory::desc blck8_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
    dnnl::memory::desc blck8_permCD_crop_tdesc = blck8_permCD_tdesc.submemory_desc(crop_dims, crop_off);
    ASSERT_TRUE(DnnlExtensionUtils::makeDescriptor(blck8_crop_tdesc)->hasLayoutType(LayoutType::nCsp8c));
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(blck8_permCD_crop_tdesc)->hasLayoutType(LayoutType::nCsp8c));
}

TEST(MemDescTest, isTailCCheck) {
    const auto dims = dnnl::memory::dims {3, 2, 5, 7};
    const auto type = dnnl::memory::data_type::u8;

    dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
    dnnl::memory::desc tailc_tdesc {dims, type, dnnl::memory::format_tag::acdb};
    dnnl::memory::desc permt_tdesc {dims, type, dnnl::memory::format_tag::bcda};
    dnnl::memory::desc blck8_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(plain_tdesc)->hasLayoutType(LayoutType::nspc));
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(permt_tdesc)->hasLayoutType(LayoutType::nspc));
    ASSERT_TRUE(DnnlExtensionUtils::makeDescriptor(tailc_tdesc)->hasLayoutType(LayoutType::nspc));
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(blck8_tdesc)->hasLayoutType(LayoutType::nspc));

    dnnl::memory::desc blck8_permCD_tdesc {dims, type, dnnl::memory::format_tag::aBdc16b};
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(blck8_permCD_tdesc)->hasLayoutType(LayoutType::nspc));

    const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
    const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};
    dnnl::memory::desc tailc_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
    ASSERT_FALSE(DnnlExtensionUtils::makeDescriptor(tailc_crop_tdesc)->hasLayoutType(LayoutType::nspc));
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

TEST(MemDescTest, KeepOrder) {
    using dnnl::memory;
    Shape dims(VectorDims{7, 3, 1, 5});
    memory::data_type dataType = memory::data_type::u8;
    DnnlBlockedMemoryDesc descPalanar(DnnlExtensionUtils::DataTypeToElementType(dataType), dims);
    ASSERT_THAT(descPalanar.getOrder(), ElementsAre(0, 1, 2, 3));

    DnnlBlockedMemoryDesc descTailC(dims, dataType, memory::format_tag::acdb);
    ASSERT_THAT(descTailC.getOrder(), ElementsAre(0, 2, 3, 1));

    DnnlBlockedMemoryDesc descBlockedC(dims, dataType, memory::format_tag::aBcd16b);
    ASSERT_THAT(descBlockedC.getOrder(), ElementsAre(0, 1, 2, 3, 1));

    DnnlBlockedMemoryDesc descWeightBlocked(dims, dataType, memory::format_tag::ABcd16b16a2b);
    ASSERT_THAT(descWeightBlocked.getOrder(), ElementsAre(0, 1, 2, 3, 1, 0, 1));

    auto dnnDims = DnnlExtensionUtils::convertToDnnlDims(dims.getStaticDims());

    memory::desc dnnlDescPlanar(dnnDims, dataType, memory::format_tag::abcd);
    ASSERT_THAT(DnnlExtensionUtils::makeDescriptor(dnnlDescPlanar)->as<DnnlBlockedMemoryDesc>()->getOrder(), ElementsAre(0, 1, 2, 3));

    memory::desc dnnlDescTailC(dnnDims, dataType, memory::format_tag::acdb);
    ASSERT_THAT(DnnlExtensionUtils::makeDescriptor(dnnlDescTailC)->as<DnnlBlockedMemoryDesc>()->getOrder(), ElementsAre(0, 2, 3, 1));

    memory::desc dnnlDescBlockedC(dnnDims, dataType, memory::format_tag::aBcd16b);
    ASSERT_THAT(DnnlExtensionUtils::makeDescriptor(dnnlDescBlockedC)->as<DnnlBlockedMemoryDesc>()->getOrder(), ElementsAre(0, 1, 2, 3, 1));

    memory::desc dnnlDescWeightBlocked(dnnDims, dataType, memory::format_tag::ABcd16b16a2b);
    ASSERT_THAT(DnnlExtensionUtils::makeDescriptor(dnnlDescWeightBlocked)->as<DnnlBlockedMemoryDesc>()->getOrder(), ElementsAre(0, 1, 2, 3, 1, 0, 1));
}

TEST(MemDescTest, UndefinedState) {
    ov::PartialShape ngraphShape({{16}, {-1, -1}, {20, 30}, {7}});
    ov::intel_cpu::Shape pluginShape(ngraphShape);
    DnnlBlockedMemoryDesc memDesc(pluginShape, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nChw8c);

    ASSERT_FALSE(memDesc.isDefined());

    ASSERT_THROW(memDesc.cloneWithNewDims({16, 7, 40, 7}), ov::Exception);
    ASSERT_THROW(memDesc.cloneWithNewDims({16, 7, 25}), ov::Exception);
    ASSERT_THROW(memDesc.cloneWithNewDims({16, 7, 25, 5}), ov::Exception);

    auto definedDesc = memDesc.cloneWithNewDims({16, 15, 25, 7});

    ASSERT_TRUE(definedDesc->isDefined());

    auto creator = BlockedDescCreator::getCommonCreators().at(LayoutType::nCsp8c);
    auto cpuBlockedDesc = creator->createSharedDesc(ov::element::f32, pluginShape);

    ASSERT_FALSE(cpuBlockedDesc->isDefined());

    ASSERT_TRUE(cpuBlockedDesc->isCompatible(memDesc));

    ASSERT_THROW(cpuBlockedDesc->cloneWithNewDims({16, 7, 40, 7}), ov::Exception);
    ASSERT_THROW(cpuBlockedDesc->cloneWithNewDims({16, 7, 25}), ov::Exception);
    ASSERT_THROW(cpuBlockedDesc->cloneWithNewDims({16, 7, 25, 5}), ov::Exception);

    auto definedBlockedDesc = cpuBlockedDesc->cloneWithNewDims({16, 15, 25, 7});

    ASSERT_TRUE(definedBlockedDesc->isDefined());

    ASSERT_FALSE(memDesc.isCompatible(*definedDesc));
    ASSERT_FALSE(memDesc.isCompatible(*definedBlockedDesc));

    ASSERT_TRUE(definedBlockedDesc->isCompatible(*definedDesc));
}

TEST(MemDescTest, MemSize) {
    constexpr size_t undefSize = MemoryDesc::UNDEFINED_SIZE;
    static const auto dnnlDataType = dnnl::memory::data_type::f32;
    static const ov::element::Type iePrc = ov::element::f32;


    ov::PartialShape ngraphShapeUndef({{16}, {-1, -1}, {20, 30}, {7}});
    ov::intel_cpu::Shape pluginShapeUndef(ngraphShapeUndef);

    auto creator = BlockedDescCreator::getCommonCreators().at(LayoutType::nspc);
    auto blockedDescUndef = creator->createDesc(iePrc, pluginShapeUndef);

    ASSERT_EQ(blockedDescUndef.getCurrentMemSize(), undefSize);
    ASSERT_EQ(blockedDescUndef.getMaxMemSize(), undefSize);

    DnnlBlockedMemoryDesc memDescUndef(pluginShapeUndef, dnnlDataType, dnnl::memory::format_tag::nhwc);

    ASSERT_EQ(memDescUndef.getCurrentMemSize(), undefSize);
    ASSERT_EQ(memDescUndef.getMaxMemSize(), undefSize);

    ov::PartialShape ngraphShapeDefUpperBound({{16}, {7, 14}, {20, 30}, {7}});
    ov::intel_cpu::Shape pluginShapeDefUpperBound(ngraphShapeDefUpperBound);

    auto blockedDescDefUpper = creator->createDesc(iePrc, pluginShapeDefUpperBound);

    ASSERT_EQ(blockedDescDefUpper.getCurrentMemSize(), undefSize);
    auto maxElementsCount = std::accumulate(pluginShapeDefUpperBound.getMaxDims().begin(),
                                            pluginShapeDefUpperBound.getMaxDims().end(),
                                            1, std::multiplies<>());
    ASSERT_EQ(blockedDescDefUpper.getMaxMemSize(), maxElementsCount * iePrc.size());

    DnnlBlockedMemoryDesc memDescDefUpper(pluginShapeDefUpperBound, dnnlDataType, dnnl::memory::format_tag::nhwc);

    ASSERT_EQ(memDescDefUpper.getCurrentMemSize(), undefSize);
    ASSERT_EQ(memDescDefUpper.getMaxMemSize(), maxElementsCount * DnnlExtensionUtils::sizeOfDataType(dnnlDataType));

    ov::PartialShape ngraphShapeDefined({{16}, {16}, {10}, {7}});
    ov::intel_cpu::Shape pluginShapeDefined(ngraphShapeDefined);

    auto blockedDescDefined = creator->createDesc(iePrc, pluginShapeDefined);

    ASSERT_NE(blockedDescDefined.getCurrentMemSize(), undefSize);
    ASSERT_NE(blockedDescDefined.getMaxMemSize(), undefSize);
    ASSERT_EQ(blockedDescDefined.getCurrentMemSize(), blockedDescDefined.getMaxMemSize());

    DnnlBlockedMemoryDesc memDescDefined(pluginShapeDefined, dnnlDataType, dnnl::memory::format_tag::nhwc);

    ASSERT_NE(memDescDefined.getCurrentMemSize(), undefSize);
    ASSERT_NE(memDescDefined.getMaxMemSize(), undefSize);
    ASSERT_EQ(memDescDefined.getCurrentMemSize(), memDescDefined.getMaxMemSize());
    ASSERT_EQ(blockedDescDefined.getCurrentMemSize(), memDescDefined.getCurrentMemSize());
}

TEST(MakeUndefinedDnnlDesc, wrongType) {
    GTEST_SKIP();
}

TEST(MakeUndefinedDnnlDesc, checkRank) {
    using dnnl::memory;
    const memory::data_type dataType = memory::data_type::u8;
    const memory::desc origin({10, 20, 15, 7}, dataType, memory::format_tag::nChw16c);

    ov::intel_cpu::Shape pluginShapeWrongRank(ov::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}});
    ASSERT_THROW(DnnlExtensionUtils::makeUndefinedDesc(origin, pluginShapeWrongRank), ov::Exception);

    ov::intel_cpu::Shape pluginShapeRightRank(ov::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}});
    MemoryDescPtr memDesc;
    OV_ASSERT_NO_THROW(memDesc = DnnlExtensionUtils::makeUndefinedDesc(origin, pluginShapeRightRank));
    ASSERT_FALSE(memDesc->isDefined());
}

TEST(MakeUndefinedDnnlDesc, checkDims) {
    using dnnl::memory;
    const memory::data_type dataType = memory::data_type::u8;
    const memory::desc origin({10, 20, 15, 7}, dataType, memory::format_tag::nChw16c);

    ov::PartialShape fullyUndef({{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}});
    for (size_t i = 0; i < fullyUndef.size(); ++i) {
        auto partialShape = fullyUndef;
        partialShape[i] = {3}; // just a number which is not equal to any origin dims
        ASSERT_THROW(DnnlExtensionUtils::makeUndefinedDesc(origin, ov::intel_cpu::Shape(partialShape)), ov::Exception);
    }

    const auto dims = origin.get_dims();
    for (size_t i = 0; i < dims.size(); ++i) {
        auto partialShape = fullyUndef;
        partialShape[i] = {dims[i]};
        MemoryDescPtr memDesc;
        OV_ASSERT_NO_THROW(memDesc = DnnlExtensionUtils::makeUndefinedDesc(origin, ov::intel_cpu::Shape(fullyUndef)));
        ASSERT_FALSE(memDesc->isDefined());
    }
}

TEST(MakeUndefinedDnnlDesc, checkLayout) {
    using dnnl::memory;
    using payloadArgs = std::tuple<memory::format_tag, memory::dims, std::string>;
    const memory::data_type dataType = memory::data_type::u8;

    payloadArgs payload[] {
            payloadArgs{ memory::format_tag::nChw16c,     {1, 1, 10, 10}, "aBcd16b" },  // auto blocked
            payloadArgs{ memory::format_tag::nhwc,        {4, 2, 10, 7 }, "acdb" },  // permuted
            payloadArgs{ memory::format_tag::nchw,        {4, 2, 10, 7 }, "abcd" },  // plain
            payloadArgs{ memory::format_tag::NChw16n16c,  {4, 2, 10, 7 }, "ABcd16a16b" },  // blocked for 2 dims
            payloadArgs{ memory::format_tag::Acdb16a,     {96, 1, 7, 7 }, "Acdb16a" },  // same strides but not default order
            payloadArgs{ memory::format_tag::BAcd16a16b,  {17, 2, 10, 7 }, "BAcd16a16b" },  // blocked and permuted outer dims
    };

    ov::PartialShape fullyUndef({{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}});

    for (const auto& item : payload) {
        dnnl::memory::format_tag fmt;
        dnnl::memory::dims dims;
        std::string strFormat;
        std::tie(fmt, dims, strFormat) = item;
        const memory::desc origin(dims, dataType, fmt);

        auto undefDesc = DnnlExtensionUtils::makeUndefinedDesc(origin, ov::intel_cpu::Shape(fullyUndef));
        ASSERT_FALSE(undefDesc->isDefined());
        ov::intel_cpu::DnnlBlockedMemoryDesc referenceDesc(ov::intel_cpu::Shape(fullyUndef), dataType, fmt);
        ASSERT_TRUE(undefDesc->isCompatible(referenceDesc));
        ASSERT_EQ(undefDesc->serializeFormat(), strFormat);
        auto defDesc = undefDesc->cloneWithNewDims(DnnlExtensionUtils::convertToVectorDims(dims));
        ASSERT_TRUE(defDesc->isDefined());
        ASSERT_EQ(origin, defDesc->as<DnnlBlockedMemoryDesc>()->getDnnlDesc());
    }
}

TEST(MakeUndefinedDnnlDesc, extraData) {
    using dnnl::memory;
    using payloadArgs = std::tuple<memory::format_tag, memory::dims>;
    const memory::data_type dataType = memory::data_type::u8;

    payloadArgs payload[] {
            payloadArgs{ memory::format_tag::nChw16c,     {1, 1, 10, 10} },  // auto blocked
            payloadArgs{ memory::format_tag::nhwc,        {4, 2, 10, 7 } },  // permuted
            payloadArgs{ memory::format_tag::nchw,        {4, 2, 10, 7 } },  // plain
            payloadArgs{ memory::format_tag::NChw16n16c,  {4, 2, 10, 7 } },  // blocked for 2 dims
            payloadArgs{ memory::format_tag::Acdb16a,     {96, 1, 7, 7 } },  // same strides but not default order
            payloadArgs{ memory::format_tag::BAcd16a16b,  {17, 2, 10, 7 } },  // blocked and permuted outer dims
    };

    ov::PartialShape fullyUndef({{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}});

    for (const auto& item : payload) {
        dnnl::memory::format_tag fmt;
        dnnl::memory::dims dims;
        std::tie(fmt, dims) = item;
        memory::desc origin(dims, dataType, fmt);

        origin.get()->extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        origin.get()->extra.compensation_mask = 1;
        origin.get()->extra.scale_adjust = 2.0f;

        auto undefDesc = DnnlExtensionUtils::makeUndefinedDesc(origin, ov::intel_cpu::Shape(fullyUndef));
        ASSERT_FALSE(undefDesc->isDefined());
        auto defDesc = undefDesc->cloneWithNewDims(DnnlExtensionUtils::convertToVectorDims(dims));
        ASSERT_TRUE(defDesc->isDefined());
        auto referenceDesc = DnnlExtensionUtils::makeDescriptor(origin);
        ASSERT_TRUE(defDesc->isCompatible(*referenceDesc));
        ASSERT_EQ(origin, defDesc->as<DnnlBlockedMemoryDesc>()->getDnnlDesc());
    }
}


TEST(isSameMethodTest, CheckTensorWithSameStrides) {
    auto isSameDataFormat = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
        dnnl::memory::desc oneDnnDesc {dims, dnnl::memory::data_type::u8, fmt};
        auto pluginDesc = DnnlExtensionUtils::makeDescriptor(oneDnnDesc);
        return pluginDesc->isSame(fmt);
    };

    std::pair<dnnl::memory::format_tag, dnnl::memory::dims> testCases[] {
        { dnnl::memory::format_tag::ntc, {1, 10, 10} },
    };

    for (const auto &tc : testCases)
        ASSERT_TRUE(isSameDataFormat(tc.first, tc.second));
}

TEST(makeDummyDesc, LowerBoundMoreThanDummyValue) {
    Shape shape(ov::PartialShape{1, 3, 85, {144, 1444}});
    auto desc = std::make_shared<DnnlBlockedMemoryDesc>(shape, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
    ASSERT_FALSE(desc->isDefined());

    MemoryDescPtr definedDesc;
    OV_ASSERT_NO_THROW(definedDesc = MemoryDescUtils::makeDummyDesc(*desc));

    ASSERT_TRUE(definedDesc->isDefined());
    ASSERT_EQ((VectorDims{1, 3, 85, 144}), definedDesc->getShape().getStaticDims());
}
