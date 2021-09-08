// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "mkldnn_memory.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/blocked_desc_creator.h"
#include "mkldnn_extension_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace testing;

TEST(MemDescTest, Conversion) {
    // Check if conversion keep desc structure
    // dnnl::memory::desc -> DnnlBlockedMemoryDesc -> CpuBlockedMemoryDesc -> DnnlBlockedMemoryDesc -> dnnl::memory::desc
    auto converted_correctly = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
        dnnl::memory::desc orig_tdesc {dims, dnnl::memory::data_type::u8, fmt};
        DnnlMemoryDescPtr plg_tdesc = MKLDNNExtensionUtils::makeDescriptor(orig_tdesc);
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

TEST(MemDescTest, ConversionTensorDesc) {
    auto converted_correctly_CpuBlockedMemoryDesc = [] (Layout layout, const SizeVector& tdDims) {
        InferenceEngine::TensorDesc orig_tdesc(InferenceEngine::Precision::FP32, tdDims, layout);

        CpuBlockedMemoryDesc cpu_blk_desc = MemoryDescUtils::convertToCpuBlockedMemoryDesc(orig_tdesc);
        InferenceEngine::TensorDesc after_tdesc = MemoryDescUtils::convertToTensorDesc(cpu_blk_desc);

        ASSERT_TRUE(cpu_blk_desc.isDefined());
        ASSERT_EQ(orig_tdesc, after_tdesc);
    };

    auto converted_correctly_DnnlBlockedMemoryDesc = [] (Layout layout, const SizeVector& tdDims) {
        InferenceEngine::TensorDesc orig_tdesc(InferenceEngine::Precision::FP32, tdDims, layout);

        DnnlBlockedMemoryDesc dnnl_blk_desc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(orig_tdesc);
        InferenceEngine::TensorDesc after_tdesc = MemoryDescUtils::convertToTensorDesc(dnnl_blk_desc);

        ASSERT_TRUE(dnnl_blk_desc.isDefined());
        ASSERT_EQ(orig_tdesc, after_tdesc);
    };

    ASSERT_ANY_THROW(converted_correctly_CpuBlockedMemoryDesc(InferenceEngine::Layout::ANY, {2, 12, 7, 7 }));
    ASSERT_ANY_THROW(converted_correctly_DnnlBlockedMemoryDesc(InferenceEngine::Layout::ANY, {2, 12, 7, 7 }));

    std::pair<Layout, SizeVector> payload[] {
            { InferenceEngine::Layout::C,            {4 } },
            { InferenceEngine::Layout::CN,           {4, 2 } },
            { InferenceEngine::Layout::NC,           {4, 2 } },
            { InferenceEngine::Layout::CHW,          {4, 2, 7 } },
            { InferenceEngine::Layout::HWC,          {4, 2, 7 } },
            { InferenceEngine::Layout::NCHW,         {4, 2, 10, 7 } },
            { InferenceEngine::Layout::NHWC,         {4, 2, 10, 7 } },
            { InferenceEngine::Layout::NCDHW,        {4, 2, 2, 10, 7} },
            { InferenceEngine::Layout::NDHWC,        {4, 2, 2, 10, 7} },
            { InferenceEngine::Layout::BLOCKED,      {4, 2, 2, 10, 7, 8} }
    };
    for (const auto &p : payload) {
        converted_correctly_CpuBlockedMemoryDesc(p.first, p.second);
        converted_correctly_DnnlBlockedMemoryDesc(p.first, p.second);
    }
}

TEST(MemDescTest, ConversionSpecialCases) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});

    auto convertToDnnlMemoryDesc = [] (dnnl::memory::format_tag fmt, const Shape& shape) {
        const DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::u8, fmt);

        // CpuBlockedMemoryDesc
        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                 dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                 dnnl_blk_desc.getOffsetPadding(), dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                 dnnl_blk_desc.getStrides());
        DnnlMemoryDescPtr dnnl_desc_ptr = MemoryDescUtils::convertToDnnlMemoryDesc(cpu_blk_desc_ptr);

        if (shape.isDynamic()) {
            ASSERT_FALSE(dnnl_desc_ptr->isDefined());
        } else {
            ASSERT_TRUE(dnnl_desc_ptr->isDefined());
        }
        ASSERT_TRUE(dnnl_blk_desc.getDnnlDesc() == dnnl_desc_ptr->getDnnlDesc());

        // DnnlMemoryDesc
        dnnl::memory::desc orig_desc {MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        orig_desc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;

        if (shape.isStatic()) {
            MemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            dnnl_desc_ptr = MemoryDescUtils::convertToDnnlMemoryDesc(desc_ptr);
            ASSERT_TRUE(dnnl_desc_ptr->isDefined());
            ASSERT_TRUE(orig_desc == dnnl_desc_ptr->getDnnlDesc());
        } else {
            ASSERT_ANY_THROW(MemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }

        // DnnlBlockedMemoryDesc
        MemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(dnnl_blk_desc);
        dnnl_desc_ptr = MemoryDescUtils::convertToDnnlMemoryDesc(dnnl_blk_desc_ptr);

        if (shape.isDynamic()) {
            ASSERT_FALSE(dnnl_desc_ptr->isDefined());
        } else {
            ASSERT_TRUE(dnnl_desc_ptr->isDefined());
        }
        ASSERT_TRUE(dnnl_blk_desc.getDnnlDesc() == dnnl_desc_ptr->getDnnlDesc());

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            dnnl_desc_ptr = MemoryDescUtils::convertToDnnlMemoryDesc(desc_extra_ptr);
            ASSERT_TRUE(dnnl_desc_ptr->getDnnlDesc() == orig_desc_extra);
        }
    };

    auto convertToDnnlBlockedMemoryDesc = [] (dnnl::memory::format_tag fmt, const Shape& shape) {
        const DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::u8, fmt);

        // CpuBlockedMemoryDesc
        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                 dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                 dnnl_blk_desc.getOffsetPadding(), dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                 dnnl_blk_desc.getStrides());
        DnnlBlockedMemoryDesc dnnl_blk_desc_check = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*cpu_blk_desc_ptr);

        if (shape.isDynamic()) {
            ASSERT_FALSE(dnnl_blk_desc_check.isDefined());
        } else {
            ASSERT_TRUE(dnnl_blk_desc_check.isDefined());
        }
        ASSERT_TRUE(dnnl_blk_desc.getDnnlDesc() == dnnl_blk_desc_check.getDnnlDesc());

        // DnnlMemoryDesc
        dnnl::memory::desc orig_desc {MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        orig_desc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;

        if (shape.isStatic()) {
            DnnlMemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            ASSERT_ANY_THROW(MemoryDescUtils::convertToBlockedMemoryDesc(desc_ptr));
        } else {
            ASSERT_ANY_THROW(DnnlMemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }

        // DnnlBlockedMemoryDesc
        dnnl_blk_desc_check = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(dnnl_blk_desc);

        if (shape.isDynamic()) {
            ASSERT_FALSE(dnnl_blk_desc_check.isDefined());
        } else {
            ASSERT_TRUE(dnnl_blk_desc_check.isDefined());
        }
        ASSERT_TRUE(dnnl_blk_desc.getDnnlDesc() == dnnl_blk_desc_check.getDnnlDesc());

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            dnnl_blk_desc_check = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*desc_extra_ptr);
            ASSERT_TRUE(dnnl_blk_desc_check.getDnnlDesc() == orig_desc_extra);
        }
    };

    auto convertToBlockedMemoryDesc = [] (dnnl::memory::format_tag fmt, const Shape& shape) {
        const DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::u8, fmt);

        // CpuBlockedMemoryDesc
        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                 dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                 dnnl_blk_desc.getOffsetPadding(), dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                 dnnl_blk_desc.getStrides());
        BlockedMemoryDescPtr blk_desc_ptr = MemoryDescUtils::convertToBlockedMemoryDesc(cpu_blk_desc_ptr);

        if (shape.isDynamic()) {
            ASSERT_FALSE(blk_desc_ptr->isDefined());
        } else {
            ASSERT_TRUE(blk_desc_ptr->isDefined());
        }
        ASSERT_TRUE(dnnl_blk_desc.getBlockDims() == blk_desc_ptr->getBlockDims() &&
        dnnl_blk_desc.getOrder() == blk_desc_ptr->getOrder() &&
        dnnl_blk_desc.getOffsetPadding() == blk_desc_ptr->getOffsetPadding() &&
        dnnl_blk_desc.getOffsetPaddingToData() == blk_desc_ptr->getOffsetPaddingToData() &&
        dnnl_blk_desc.getStrides() == blk_desc_ptr->getStrides());

        // DnnlMemoryDesc
        dnnl::memory::desc orig_desc {MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        orig_desc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;

        if (shape.isStatic()) {
            DnnlMemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            ASSERT_ANY_THROW(MemoryDescUtils::convertToBlockedMemoryDesc(desc_ptr));
        } else {
            ASSERT_ANY_THROW(DnnlMemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }

        // DnnlBlockedMemoryDesc
        MemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(dnnl_blk_desc);
        blk_desc_ptr = MemoryDescUtils::convertToBlockedMemoryDesc(dnnl_blk_desc_ptr);

        if (shape.isDynamic()) {
            ASSERT_FALSE(blk_desc_ptr->isDefined());
        } else {
            ASSERT_TRUE(blk_desc_ptr->isDefined());
        }
        ASSERT_TRUE(dnnl_blk_desc.getBlockDims() == blk_desc_ptr->getBlockDims() &&
        dnnl_blk_desc.getOrder() == blk_desc_ptr->getOrder() &&
        dnnl_blk_desc.getOffsetPadding() == blk_desc_ptr->getOffsetPadding() &&
        dnnl_blk_desc.getOffsetPaddingToData() == blk_desc_ptr->getOffsetPaddingToData() &&
        dnnl_blk_desc.getStrides() == blk_desc_ptr->getStrides());

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            blk_desc_ptr = MemoryDescUtils::convertToBlockedMemoryDesc(desc_extra_ptr);
            // TODO [DS]: Should be uncommented after correcting copying data.extra
            // ASSERT_TRUE(blk_desc_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
        }
    };

    std::pair<dnnl::memory::format_tag, Shape> payload[] {
            { dnnl::memory::format_tag::nChw16c,     Shape(SizeVector {1, 1, 10, 10}) },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        Shape(SizeVector {4, 2, 10, 7 }) },  // permuted
            { dnnl::memory::format_tag::nchw,        Shape(SizeVector {4, 2, 10, 7 }) },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  Shape(SizeVector {4, 2, 10, 7 }) },  // blocked for 2 dims
            { dnnl::memory::format_tag::BAcd16a16b,  Shape(SizeVector {4, 2, 10, 7 }) },  // blocked and permuted outer dims
            { dnnl::memory::format_tag::Acdb16a,     Shape(SizeVector {96, 1, 7, 7 }) },  // same strides but not default order
            { dnnl::memory::format_tag::nChw16c,     dynamicShape },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        dynamicShape },  // permuted
            { dnnl::memory::format_tag::nchw,        dynamicShape },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  dynamicShape },  // blocked for 2 dims
            { dnnl::memory::format_tag::BAcd16a16b,  dynamicShape },  // blocked and permuted outer dims
            { dnnl::memory::format_tag::Acdb16a,     dynamicShape },  // same strides but not default order
    };

    for (const auto &p : payload) {
        convertToDnnlMemoryDesc(p.first, p.second);
        convertToDnnlBlockedMemoryDesc(p.first, p.second);
        convertToBlockedMemoryDesc(p.first, p.second);
    }
}

TEST(MemDescTest, cloneWithNewPrecision) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});

    auto cloneWithNewPrecision = [] (dnnl::memory::format_tag fmt, const Shape& shape, InferenceEngine::Precision pr) {
        const DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::u8, fmt);

        // CpuBlockedMemoryDesc
        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(),
                                                                                 dnnl_blk_desc.getShape(),
                                                                                 dnnl_blk_desc.getBlockDims(),
                                                                                 dnnl_blk_desc.getOrder(),
                                                                                 dnnl_blk_desc.getOffsetPadding(),
                                                                                 dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                 dnnl_blk_desc.getStrides());
        MemoryDescPtr cloneWithNewPrec = cpu_blk_desc_ptr->cloneWithNewPrecision(pr);
        auto desc = (*cloneWithNewPrec).as<CpuBlockedMemoryDesc>();

        if (shape.isDynamic()) {
            ASSERT_FALSE(cloneWithNewPrec->isDefined());
        } else {
            ASSERT_TRUE(cloneWithNewPrec->isDefined());
        }
        ASSERT_TRUE(cloneWithNewPrec->getPrecision() == pr &&
                    dnnl_blk_desc.getBlockDims() == desc->getBlockDims() &&
                    dnnl_blk_desc.getOrder() == desc->getOrder() &&
                    dnnl_blk_desc.getOffsetPadding() == desc->getOffsetPadding() &&
                    dnnl_blk_desc.getOffsetPaddingToData() == desc->getOffsetPaddingToData() &&
                    dnnl_blk_desc.getStrides() == desc->getStrides());



        // DnnlMemoryDesc
        dnnl::memory::desc orig_desc{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        orig_desc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;

        if (shape.isStatic()) {
            MemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            cloneWithNewPrec = desc_ptr->cloneWithNewPrecision(pr);
            dnnl::memory::desc orig_desc_new_pr{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()),
                                                MKLDNNExtensionUtils::IEPrecisionToDataType(pr), fmt};
            orig_desc_new_pr.data.format_kind = dnnl_format_kind_undef;
            orig_desc_new_pr.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            ASSERT_TRUE(cloneWithNewPrec->getPrecision() == pr &&
                        cloneWithNewPrec->as<DnnlMemoryDesc>()->getDnnlDesc() == orig_desc_new_pr);
        } else {
            ASSERT_ANY_THROW(MemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }

        // DnnlBlockedMemoryDesc
        cloneWithNewPrec = dnnl_blk_desc.cloneWithNewPrecision(pr);
        DnnlBlockedMemoryDesc dnnl_blk_desc_new_pr(shape, MKLDNNExtensionUtils::IEPrecisionToDataType(pr), fmt);

        if (shape.isDynamic()) {
            ASSERT_FALSE(cloneWithNewPrec->isDefined());
        } else {
            ASSERT_TRUE(cloneWithNewPrec->isDefined());
        }
        ASSERT_TRUE(cloneWithNewPrec->getPrecision() == pr &&
                    dnnl_blk_desc.getBlockDims() == dnnl_blk_desc_new_pr.getBlockDims() &&
                    dnnl_blk_desc.getOrder() == dnnl_blk_desc_new_pr.getOrder() &&
                    dnnl_blk_desc.getOffsetPadding() == dnnl_blk_desc_new_pr.getOffsetPadding() &&
                    dnnl_blk_desc.getOffsetPaddingToData() == dnnl_blk_desc_new_pr.getOffsetPaddingToData() &&
                    dnnl_blk_desc.getStrides() == dnnl_blk_desc_new_pr.getStrides());

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            MemoryDescPtr clone = desc_extra_ptr->cloneWithNewPrecision(pr);
            // TODO [DS]: Should be uncommented after correcting copying data.extra
            // ASSERT_TRUE(clone->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
        }
    };

    std::pair<dnnl::memory::format_tag, Shape> payload[] {
            { dnnl::memory::format_tag::nChw16c,     Shape(SizeVector {1, 1, 10, 10}) },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        Shape(SizeVector {4, 2, 10, 7 }) },  // permuted
            { dnnl::memory::format_tag::nchw,        Shape(SizeVector {4, 2, 10, 7 }) },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  Shape(SizeVector {4, 2, 10, 7 }) },  // blocked for 2 dims
            { dnnl::memory::format_tag::BAcd16a16b,  Shape(SizeVector {4, 2, 10, 7 }) },  // blocked and permuted outer dims
            { dnnl::memory::format_tag::Acdb16a,     Shape(SizeVector {96, 1, 7, 7 }) },  // same strides but not default order
            { dnnl::memory::format_tag::nChw16c,     dynamicShape },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        dynamicShape },  // permuted
            { dnnl::memory::format_tag::nchw,        dynamicShape },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  dynamicShape },  // blocked for 2 dims
            { dnnl::memory::format_tag::BAcd16a16b,  dynamicShape },  // blocked and permuted outer dims
            { dnnl::memory::format_tag::Acdb16a,     dynamicShape },  // same strides but not default order
    };

    std::vector<InferenceEngine::Precision> precision = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::BF16,
            InferenceEngine::Precision::I8,
            InferenceEngine::Precision::I32
    };

    for (const auto &p : payload) {
        for (const auto &pr : precision) {
            cloneWithNewPrecision(p.first, p.second, pr);
        }
    }
}

TEST(MemDescTest, convertToTensorDesc) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});

    auto convertToTensorDesc = [] (dnnl::memory::format_tag fmt, const Shape& shape) {
        const DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::u8, fmt);
        InferenceEngine::TensorDesc orig_tdesc(InferenceEngine::Precision::U8, shape.getDims(), BlockingDesc{dnnl_blk_desc.getBlockDims(),
                                                                                                             dnnl_blk_desc.getOrder(),
                                                                                                             dnnl_blk_desc.getOffsetPadding(),
                                                                                                             dnnl_blk_desc.getOffsetPaddingToData()});

        // CpuBlockedMemoryDesc
        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                 dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                 dnnl_blk_desc.getOffsetPadding(), dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                 dnnl_blk_desc.getStrides());
        InferenceEngine::TensorDesc tensor_desc = MemoryDescUtils::convertToTensorDesc(*cpu_blk_desc_ptr);

        ASSERT_TRUE(orig_tdesc == tensor_desc);

        // DnnlBlockedMemoryDesc
        MemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(dnnl_blk_desc);
        tensor_desc = MemoryDescUtils::convertToTensorDesc(*dnnl_blk_desc_ptr);

        ASSERT_TRUE(orig_tdesc == tensor_desc);

        // DnnlMemoryDesc
        dnnl::memory::desc orig_desc {MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        orig_desc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        MemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);

        ASSERT_ANY_THROW(tensor_desc = MemoryDescUtils::convertToTensorDesc(*desc_ptr));

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            ASSERT_ANY_THROW(tensor_desc = MemoryDescUtils::convertToTensorDesc(*desc_extra_ptr));
        }
    };

    std::pair<dnnl::memory::format_tag, Shape> payload[] {
            { dnnl::memory::format_tag::nChw16c,     Shape(SizeVector {1, 1, 10, 10}) },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        Shape(SizeVector {4, 2, 10, 7 }) },  // permuted
            { dnnl::memory::format_tag::nchw,        Shape(SizeVector {4, 2, 10, 7 }) },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  Shape(SizeVector {4, 2, 10, 7 }) },  // blocked for 2 dims
            { dnnl::memory::format_tag::Acdb16a,     Shape(SizeVector {96, 1, 7, 7 }) },  // same strides but not default order
    };

    for (const auto &p : payload) {
        convertToTensorDesc(p.first, p.second);
    }
    ASSERT_ANY_THROW(convertToTensorDesc(dnnl::memory::format_tag::nChw16c, dynamicShape));
}

TEST(MemDescTest, isCompatibleTrueTests) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});

    auto isCompatibleDnnlMemoryDesc = [] (dnnl::memory::format_tag fmt, const Shape& shape) {
        dnnl::memory::desc orig_desc{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;

        if (shape.isStatic()) {
            MemoryDescPtr dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            ASSERT_TRUE(dnnl_desc_ptr->isCompatible(*dnnl_desc_ptr));
        } else {
            ASSERT_ANY_THROW(MemoryDescPtr dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }
    };

    auto isCompatibleDnnlBlockedMemoryDesc = [] (dnnl::memory::format_tag fmt, const Shape& shape) {
        DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::u8, fmt);

        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                dnnl_blk_desc.getOffsetPadding(), dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                dnnl_blk_desc.getStrides());

        ASSERT_TRUE(dnnl_blk_desc.isCompatible(dnnl_blk_desc) &&
               dnnl_blk_desc.isCompatible(*cpu_blk_desc_ptr));

        dnnl::memory::desc orig_desc{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        MemoryDescPtr dnnl_desc_ptr;

        if (shape.isStatic()) {
            dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            ASSERT_FALSE(dnnl_blk_desc.isCompatible(*dnnl_desc_ptr));
        } else {
            ASSERT_ANY_THROW(dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            ASSERT_FALSE(desc_extra_ptr->isCompatible(dnnl_blk_desc));
        }
    };

    auto isCompatibleCpuBlockedMemoryDesc = [] (dnnl::memory::format_tag fmt, const Shape& shape) {
        DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::u8, fmt);

        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                dnnl_blk_desc.getOffsetPadding(), dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                dnnl_blk_desc.getStrides());

        ASSERT_TRUE(cpu_blk_desc_ptr->isCompatible(*cpu_blk_desc_ptr) &&
                    cpu_blk_desc_ptr->isCompatible(*cpu_blk_desc_ptr));

        dnnl::memory::desc orig_desc{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        MemoryDescPtr dnnl_desc_ptr;
        orig_desc.data.format_kind = dnnl_format_kind_undef;

        if (shape.isStatic()) {
            dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            ASSERT_FALSE(dnnl_blk_desc.isCompatible(*dnnl_desc_ptr));
        } else {
            ASSERT_ANY_THROW(dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }
    };

    std::pair<dnnl::memory::format_tag, Shape> payload[] {
            { dnnl::memory::format_tag::nChw16c,     Shape(SizeVector {1, 1, 10, 10}) },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        Shape(SizeVector {4, 2, 10, 7 }) },  // permuted
            { dnnl::memory::format_tag::nchw,        Shape(SizeVector {4, 2, 10, 7 }) },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  Shape(SizeVector {4, 2, 10, 7 }) },  // blocked for 2 dims
            { dnnl::memory::format_tag::BAcd16a16b,  Shape(SizeVector {4, 2, 10, 7 }) },  // blocked and permuted outer dims
            { dnnl::memory::format_tag::Acdb16a,     Shape(SizeVector {96, 1, 7, 7 }) },  // same strides but not default order
            { dnnl::memory::format_tag::nChw16c,     dynamicShape },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        dynamicShape },  // permuted
            { dnnl::memory::format_tag::nchw,        dynamicShape },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  dynamicShape },  // blocked for 2 dims
            { dnnl::memory::format_tag::BAcd16a16b,  dynamicShape },  // blocked and permuted outer dims
            { dnnl::memory::format_tag::Acdb16a,     dynamicShape },  // same strides but not default order
    };

    for (const auto &p : payload) {
        isCompatibleDnnlMemoryDesc(p.first, p.second);
        isCompatibleDnnlBlockedMemoryDesc(p.first, p.second);
        isCompatibleCpuBlockedMemoryDesc(p.first, p.second);
    }

    // Should be True after fix
    auto isCompatibleEquallyStoredInMemory = [] (dnnl::memory::format_tag fmt1, dnnl::memory::format_tag fmt2, const Shape& shape) {
        DnnlBlockedMemoryDesc blk_desc1(shape, dnnl::memory::data_type::u8, fmt1);
        MemoryDescPtr cpu_blk_desc_ptr_ncsp = std::make_shared<CpuBlockedMemoryDesc>(blk_desc1.getPrecision(), blk_desc1.getShape(),
                                                                                     blk_desc1.getBlockDims(), blk_desc1.getOrder(),
                                                                                     blk_desc1.getOffsetPadding(), blk_desc1.getOffsetPaddingToData(),
                                                                                     blk_desc1.getStrides());

        DnnlBlockedMemoryDesc blk_desc2(shape, dnnl::memory::data_type::u8, fmt2);
        MemoryDescPtr cpu_blk_desc_ptr_nspc = std::make_shared<CpuBlockedMemoryDesc>(blk_desc2.getPrecision(), blk_desc2.getShape(),
                                                                                     blk_desc2.getBlockDims(), blk_desc2.getOrder(),
                                                                                     blk_desc2.getOffsetPadding(), blk_desc2.getOffsetPaddingToData(),
                                                                                     blk_desc2.getStrides());

        return ((*cpu_blk_desc_ptr_ncsp).isCompatible(*cpu_blk_desc_ptr_nspc));
    };

    ASSERT_FALSE(isCompatibleEquallyStoredInMemory(dnnl::memory::format_tag::nchw, dnnl::memory::format_tag::nhwc,
                                                   Shape(SizeVector{1, 3, 1, 1})));
    ASSERT_FALSE(isCompatibleEquallyStoredInMemory(dnnl::memory::format_tag::nchw, dnnl::memory::format_tag::nhwc,
                                                   Shape(SizeVector{1, 1, 2, 3})));
    ASSERT_FALSE(isCompatibleEquallyStoredInMemory(dnnl::memory::format_tag::nchw, dnnl::memory::format_tag::nhwc,
                                                   Shape(ngraph::PartialShape{{1}, {1, 3}, {1, 2}, {1, 3}})));
    ASSERT_FALSE(isCompatibleEquallyStoredInMemory(dnnl::memory::format_tag::abc, dnnl::memory::format_tag::bac,
                                                   Shape(ngraph::PartialShape{{1}, {2, 3}, {4, 5}})));
}

TEST(MemDescTest, isCompatibleFalseTests) {
    auto isCompatibleCpuBlockedMemoryDescExtraFlags = [] (dnnl::memory::format_tag fmt, const dnnl::memory::dims& dims) {
        dnnl::memory::desc orig_desc_extra {dims, dnnl::memory::data_type::u8, fmt};
        orig_desc_extra.data.format_kind = dnnl_format_kind_undef;
        orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        dnnl::memory::desc orig_desc {dims, dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;

        // DnnlMemoryDesc
        DnnlMemoryDescPtr dnnl_desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
        DnnlMemoryDescPtr dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);

        ASSERT_FALSE(dnnl_desc_ptr->isCompatible(*dnnl_desc_extra_ptr));

        // DnnlBlockedMemoryDesc
        DnnlBlockedMemoryDesc dnnl_blk_desc(dnnl_desc_ptr->getShape(), dnnl::memory::data_type::u8, fmt);

        ASSERT_FALSE(dnnl_blk_desc.isCompatible(*dnnl_desc_extra_ptr));

        // CpuBlockedMemoryDesc
        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                dnnl_blk_desc.getOffsetPadding(), dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                dnnl_blk_desc.getStrides());

        ASSERT_FALSE(cpu_blk_desc_ptr->isCompatible(*dnnl_desc_extra_ptr));

        // DnnlBlockedMemoryDesc with extra data
        dnnl::memory::desc desc_extra {dims, dnnl::memory::data_type::u8, fmt};
        desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        desc_extra.data.extra.compensation_mask = 1;
        desc_extra.data.extra.scale_adjust = 2.0f;
        dnnl::memory::desc desc {dims, dnnl::memory::data_type::u8, fmt};
        MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(desc_extra);
        MemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(desc);

        ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == desc_extra);
        ASSERT_FALSE(desc_extra_ptr->isCompatible(*desc_ptr));
    };

    auto isCompatibleCpuBlockedMemoryDescDifferentShapes  = [] (dnnl::memory::format_tag fmt, const dnnl::memory::dims& dims) {
        dnnl::memory::dims newDims{1, 1, 1, 1};
        dnnl::memory::desc orig_desc {dims, dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        dnnl::memory::desc descNewDims {newDims, dnnl::memory::data_type::u8, fmt};
        descNewDims.data.format_kind = dnnl_format_kind_undef;

        // DnnlMemoryDesc
        DnnlMemoryDescPtr dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
        DnnlMemoryDescPtr dnnl_desc_new_dims = MKLDNNExtensionUtils::makeDescriptor(descNewDims);

        ASSERT_FALSE(dnnl_desc_ptr->isCompatible(*dnnl_desc_new_dims));

        // DnnlBlockedMemoryDesc
        DnnlBlockedMemoryDesc dnnl_blk_desc(dnnl_desc_ptr->getShape(), dnnl::memory::data_type::u8, fmt);
        DnnlBlockedMemoryDesc dnnl_blk_desc_new_dims(dnnl_desc_new_dims->getShape(), dnnl::memory::data_type::u8, fmt);

        ASSERT_FALSE(dnnl_blk_desc.isCompatible(dnnl_blk_desc_new_dims));

        // CpuBlockedMemoryDesc
        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                dnnl_blk_desc.getOffsetPadding(), dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                dnnl_blk_desc.getStrides());
        MemoryDescPtr cpu_blk_desc_ptr_copy = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_new_dims.getPrecision(), dnnl_blk_desc_new_dims.getShape(),
                                                                                dnnl_blk_desc_new_dims.getBlockDims(), dnnl_blk_desc_new_dims.getOrder(),
                                                                                dnnl_blk_desc_new_dims.getOffsetPadding(),
                                                                                dnnl_blk_desc_new_dims.getOffsetPaddingToData(),
                                                                                dnnl_blk_desc_new_dims.getStrides());

        ASSERT_FALSE(cpu_blk_desc_ptr->isCompatible(*cpu_blk_desc_ptr_copy));
    };

    std::pair<dnnl::memory::format_tag, dnnl::memory::dims> payload[] {
            { dnnl::memory::format_tag::nChw16c,     {1, 1, 10, 10} },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        {4, 2, 10, 7 } },  // permuted
            { dnnl::memory::format_tag::nchw,        {4, 2, 10, 7 } },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  {4, 2, 10, 7 } },  // blocked for 2 dims
            { dnnl::memory::format_tag::BAcd16a16b,  {4, 2, 10, 7 } },  // blocked and permuted outer dims
            { dnnl::memory::format_tag::Acdb16a,     {96, 1, 7, 7 } },  // same strides but not default order
    };
    for (const auto &p : payload) {
        isCompatibleCpuBlockedMemoryDescExtraFlags(p.first, p.second);
        isCompatibleCpuBlockedMemoryDescDifferentShapes(p.first, p.second);
    }
}

TEST(MemDescTest, hasLayoutType) {
    auto hasLayoutTypeDnnlDesc = [] (const Shape& shape,
                                     const dnnl::memory::format_tag formatTag,
                                     const MKLDNNPlugin::LayoutType expected) {
        dnnl::memory::desc orig_desc {MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::f32, formatTag};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        orig_desc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;

        if (shape.isStatic()) {
            DnnlMemoryDescPtr dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            ASSERT_FALSE(dnnl_desc_ptr->hasLayoutType(expected));
        } else {
            ASSERT_ANY_THROW(DnnlMemoryDescPtr dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }
    };

    auto hasLayoutTypeDnnlBlkDesc = [] (const Shape& shape,
                                        const dnnl::memory::format_tag formatTag,
                                        const MKLDNNPlugin::LayoutType expected) {
        DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::f32, formatTag);

        ASSERT_TRUE(dnnl_blk_desc.hasLayoutType(expected));

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, formatTag};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->hasLayoutType(expected));
        }
    };

    auto hasLayoutTypeCpuBlkDesc = [] (const Shape& shape,
                                       const dnnl::memory::format_tag formatTag,
                                       const MKLDNNPlugin::LayoutType expected) {
        DnnlBlockedMemoryDesc dnnl_blk_desc(shape, dnnl::memory::data_type::f32, formatTag);

        MemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc.getPrecision(), dnnl_blk_desc.getShape(),
                                                                                dnnl_blk_desc.getBlockDims(), dnnl_blk_desc.getOrder(),
                                                                                dnnl_blk_desc.getOffsetPadding(),
                                                                                dnnl_blk_desc.getOffsetPaddingToData(),
                                                                                dnnl_blk_desc.getStrides());

        ASSERT_TRUE(cpu_blk_desc_ptr->hasLayoutType(expected));
    };

    auto hasLayoutTypeNcspNscp = [] (const Shape& shape) {
        DnnlBlockedMemoryDesc dnnl_blk_desc1(shape, dnnl::memory::data_type::f32, mkldnn::memory::format_tag::nchw);
        MemoryDescPtr cpu_blk_desc_ptr1 = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc1.getPrecision(), dnnl_blk_desc1.getShape(),
                                                                                 dnnl_blk_desc1.getBlockDims(), dnnl_blk_desc1.getOrder(),
                                                                                 dnnl_blk_desc1.getOffsetPadding(), dnnl_blk_desc1.getOffsetPaddingToData(),
                                                                                 dnnl_blk_desc1.getStrides());

        ASSERT_TRUE(dnnl_blk_desc1.hasLayoutType(MKLDNNPlugin::LayoutType::ncsp));
        ASSERT_TRUE(cpu_blk_desc_ptr1->hasLayoutType(MKLDNNPlugin::LayoutType::ncsp));
        ASSERT_FALSE(dnnl_blk_desc1.hasLayoutType(MKLDNNPlugin::LayoutType::nspc));
        ASSERT_FALSE(cpu_blk_desc_ptr1->hasLayoutType(MKLDNNPlugin::LayoutType::nspc));

        DnnlBlockedMemoryDesc dnnl_blk_desc2(shape, dnnl::memory::data_type::f32, mkldnn::memory::format_tag::nhwc);
        MemoryDescPtr cpu_blk_desc_ptr2 = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc2.getPrecision(), dnnl_blk_desc2.getShape(),
                                                                                 dnnl_blk_desc2.getBlockDims(), dnnl_blk_desc2.getOrder(),
                                                                                 dnnl_blk_desc2.getOffsetPadding(), dnnl_blk_desc2.getOffsetPaddingToData(),
                                                                                 dnnl_blk_desc2.getStrides());

        ASSERT_TRUE(dnnl_blk_desc2.hasLayoutType(MKLDNNPlugin::LayoutType::nspc));
        ASSERT_TRUE(cpu_blk_desc_ptr2->hasLayoutType(MKLDNNPlugin::LayoutType::nspc));
        ASSERT_FALSE(dnnl_blk_desc2.hasLayoutType(MKLDNNPlugin::LayoutType::ncsp));
        ASSERT_FALSE(cpu_blk_desc_ptr2->hasLayoutType(MKLDNNPlugin::LayoutType::ncsp));
    };

    std::pair<std::pair<dnnl::memory::format_tag, Shape>, MKLDNNPlugin::LayoutType> payload[] {
            {{ dnnl::memory::format_tag::nChw16c,     Shape(SizeVector{1, 1, 10, 10}) }, MKLDNNPlugin::LayoutType::nCsp16c},
            {{ dnnl::memory::format_tag::nhwc,        Shape(SizeVector{4, 2, 10, 7 }) }, MKLDNNPlugin::LayoutType::nspc},
            {{ dnnl::memory::format_tag::nchw,        Shape(SizeVector{4, 2, 10, 7 }) }, MKLDNNPlugin::LayoutType::ncsp},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(SizeVector{1, 1, 10, 10}) }, MKLDNNPlugin::LayoutType::nCsp8c},
            {{ dnnl::memory::format_tag::nChw16c,     Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}) }, MKLDNNPlugin::LayoutType::nCsp16c},
            {{ dnnl::memory::format_tag::nhwc,        Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}) }, MKLDNNPlugin::LayoutType::nspc},
            {{ dnnl::memory::format_tag::nchw,        Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}) }, MKLDNNPlugin::LayoutType::ncsp},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}) }, MKLDNNPlugin::LayoutType::nCsp8c}
    };

    for (auto& p : payload) {
        hasLayoutTypeDnnlDesc(p.first.second, p.first.first, p.second);
        hasLayoutTypeDnnlBlkDesc(p.first.second, p.first.first, p.second);
        hasLayoutTypeCpuBlkDesc(p.first.second, p.first.first, p.second);
        hasLayoutTypeNcspNscp(p.first.second);
    }
}

TEST(MemDescTest, blocksExtended) {
    auto blocksExtended = [&] (const Shape& shape, const dnnl::memory::format_tag formatTag, const bool result) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::u8, formatTag);
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());
        if (shape.isStatic()) {
            ASSERT_EQ(result, (*dnnl_blk_desc_ptr).blocksExtended());
            ASSERT_EQ(result, (*cpu_blk_desc_ptr).blocksExtended());
        } else {
            ASSERT_ANY_THROW((*dnnl_blk_desc_ptr).blocksExtended());
            ASSERT_ANY_THROW((*cpu_blk_desc_ptr).blocksExtended());
        }

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, formatTag};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            ASSERT_EQ(result, desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->blocksExtended());
        }
    };

    std::pair<std::pair<dnnl::memory::format_tag, Shape>, bool> payload[] {
            {{ dnnl::memory::format_tag::nChw16c,     Shape(SizeVector{1, 16, 10, 10}) }, false},
            {{ dnnl::memory::format_tag::nChw16c,     Shape(SizeVector{1, 1, 10, 10}) }, true},
            {{ dnnl::memory::format_tag::nhwc,        Shape(SizeVector{4, 2, 10, 7 }) }, false},
            {{ dnnl::memory::format_tag::nchw,        Shape(SizeVector{4, 2, 10, 7 }) }, false},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(SizeVector{1, 1, 10, 10}) }, true},
            {{ dnnl::memory::format_tag::nChw16c,     Shape(ngraph::PartialShape{{16}, {16}, {-1, -1}, {-1, -1}}) }, false},
            {{ dnnl::memory::format_tag::nChw16c,     Shape(ngraph::PartialShape{{16}, {6}, {-1, -1}, {-1, -1}}) }, true},
            {{ dnnl::memory::format_tag::nhwc,        Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}) }, false},
            {{ dnnl::memory::format_tag::nchw,        Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}) }, false},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}) }, false},
    };

    for (auto& p : payload) {
        blocksExtended(p.first.second, p.first.first, p.second);
    }
}

TEST(MemDescTest, getPaddedElementsCount) {
    auto getPaddedElementsStatic = [] (const Shape& shape, const dnnl::memory::format_tag formatTag, const size_t result) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::u8, formatTag);
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());

        ASSERT_EQ(result, (*dnnl_blk_desc_ptr).getPaddedElementsCount());
        ASSERT_EQ(result, (*cpu_blk_desc_ptr).getPaddedElementsCount());
    };

    auto getPaddedElementsDynamic = [] (const Shape& shape, const dnnl::memory::format_tag formatTag) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::u8, formatTag);
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());

        ASSERT_ANY_THROW((*dnnl_blk_desc_ptr).getPaddedElementsCount());
        ASSERT_ANY_THROW((*cpu_blk_desc_ptr).getPaddedElementsCount());
    };

    std::pair<std::pair<dnnl::memory::format_tag, Shape>, size_t> payload[] {
            {{ dnnl::memory::format_tag::nChw16c,     Shape(SizeVector{1, 16, 10, 10}) }, 1600},
            {{ dnnl::memory::format_tag::nChw16c,     Shape(SizeVector{1, 1, 10, 10}) }, 1600},
            {{ dnnl::memory::format_tag::nhwc,        Shape(SizeVector{4, 2, 10, 7 }) }, 560},
            {{ dnnl::memory::format_tag::nchw,        Shape(SizeVector{4, 4, 10, 7 }) }, 1120},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(SizeVector{1, 1, 10, 10}) }, 800}
    };

    for (auto& p : payload) {
        getPaddedElementsStatic(p.first.second, p.first.first, p.second);
    }
    getPaddedElementsDynamic(Shape(ngraph::PartialShape{{16}, {16}, {-1, -1}, {-1, -1}}), dnnl::memory::format_tag::nChw16c);
    getPaddedElementsDynamic(Shape(ngraph::PartialShape{{16}, {16}, {-1, -1}, {-1, -1}}), dnnl::memory::format_tag::nhwc);
}

TEST(MemDescTest, cloneWithNewDims) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});

    auto cloneWithNewDimsDnnlDesc = [&] (dnnl::memory::format_tag fmt) {
        dnnl::memory::desc orig_desc {MKLDNNExtensionUtils::convertToDnnlDims(dynamicShape.getDims()), dnnl::memory::data_type::f32, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        orig_desc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;

        ASSERT_ANY_THROW(DnnlMemoryDescPtr dnnl_desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
    };

    auto cloneWithNewDimsDnnlBlkDesc = [&] (dnnl::memory::format_tag fmt) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(dynamicShape, mkldnn::memory::data_type::u8, fmt);
        DnnlBlockedMemoryDescPtr ref = std::make_shared<DnnlBlockedMemoryDesc>(Shape(SizeVector {16, 8, 10, 10}), mkldnn::memory::data_type::u8, fmt);
        auto descNewDims = dnnl_blk_desc_ptr->cloneWithNewDims({16, 8, 10, 10});

        ASSERT_EQ(descNewDims->as<DnnlBlockedMemoryDesc>()->getDnnlDesc(), ref->getDnnlDesc());
        ASSERT_EQ(descNewDims->as<DnnlBlockedMemoryDesc>()->getStrides(), ref->getStrides());
        ASSERT_EQ(descNewDims->as<DnnlBlockedMemoryDesc>()->getOrder(), ref->getOrder());
        ASSERT_ANY_THROW(descNewDims = dnnl_blk_desc_ptr->cloneWithNewDims({8, 8, 10, 10}));
        ASSERT_ANY_THROW(descNewDims = dnnl_blk_desc_ptr->cloneWithNewDims({16, 16, 10, 10}));
    };

    auto cloneWithNewDimsCpuBlkDesc = [&] (dnnl::memory::format_tag fmt) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(dynamicShape, mkldnn::memory::data_type::u8, fmt);
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());
        auto descNewDims = cpu_blk_desc_ptr->cloneWithNewDims({16, 9, 10, 10});
        DnnlBlockedMemoryDescPtr ref = std::make_shared<DnnlBlockedMemoryDesc>(Shape(SizeVector{16, 9, 10, 10}), mkldnn::memory::data_type::u8, fmt);

        ASSERT_EQ(MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*descNewDims).getDnnlDesc(), ref->getDnnlDesc());
        ASSERT_EQ(MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*descNewDims).getStrides(), ref->getStrides());
        ASSERT_EQ(MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*descNewDims).getOrder(), ref->getOrder());
        ASSERT_ANY_THROW(descNewDims = ref->cloneWithNewDims({8, 8, 10, 10}));
        ASSERT_ANY_THROW(descNewDims = ref->cloneWithNewDims({16, 16, 10, 10}));
    };

    dnnl::memory::format_tag payload[] {
        dnnl::memory::format_tag::nChw16c,
        dnnl::memory::format_tag::nhwc,
        dnnl::memory::format_tag::nchw,
        dnnl::memory::format_tag::NChw16n16c,
        // TODO [DS]: Should be fixed for DnnlBlockedMemoryDesc
        // dnnl::memory::format_tag::BAcd16a16b,
        dnnl::memory::format_tag::Acdb16a,
    };

    for (const auto &p : payload) {
        cloneWithNewDimsDnnlDesc(p);
        cloneWithNewDimsDnnlBlkDesc(p);
        cloneWithNewDimsCpuBlkDesc(p);
    }
}

TEST(MemDescTest, isDefined) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});
    Shape staticShape(SizeVector{1, 1, 10, 10});

    auto isDefinedStaticShape = [&] (dnnl::memory::format_tag fmt) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr;

        // DnnlBlockedMemoryDesc
        dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(staticShape, mkldnn::memory::data_type::u8, fmt);

        ASSERT_TRUE(dnnl_blk_desc_ptr->isDefined());

        // CpuBlockedMemoryDesc
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());

        ASSERT_TRUE(cpu_blk_desc_ptr->isDefined());

        // DnnlBlockedMemoryDesc with extra data
        dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(staticShape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        orig_desc_extra.data.extra.compensation_mask = 1;
        orig_desc_extra.data.extra.scale_adjust = 2.0f;
        MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
        ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
        ASSERT_TRUE(desc_extra_ptr->isDefined());
    };

    auto isDefinedDynamicShape = [&] (dnnl::memory::format_tag fmt) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr;

        // DnnlBlockedMemoryDesc
        dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(dynamicShape, mkldnn::memory::data_type::u8, fmt);

        ASSERT_FALSE(dnnl_blk_desc_ptr->isDefined());

        // CpuBlockedMemoryDesc
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());

        ASSERT_FALSE(cpu_blk_desc_ptr->isDefined());
    };

    dnnl::memory::format_tag payload[] {
            dnnl::memory::format_tag::nChw16c,
            dnnl::memory::format_tag::nhwc,
            dnnl::memory::format_tag::nchw,
            dnnl::memory::format_tag::NChw16n16c,
            dnnl::memory::format_tag::BAcd16a16b,
            dnnl::memory::format_tag::Acdb16a,
    };

    for (const auto &p : payload) {
        isDefinedStaticShape(p);
        isDefinedDynamicShape(p);
    }
}

TEST(MemDescTest, getMaxMemSize) {
    auto getMaxMemSize = [&] (const Shape& shape, const dnnl::memory::format_tag formatTag, const size_t result) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::u8, formatTag);
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());

        ASSERT_EQ(result, dnnl_blk_desc_ptr->getMaxMemSize());
        ASSERT_EQ(result, cpu_blk_desc_ptr->getMaxMemSize());

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, formatTag};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            ASSERT_LT(result, desc_extra_ptr->getMaxMemSize());
        }
    };

    std::pair<std::pair<dnnl::memory::format_tag, Shape>, size_t> payload[] {
            {{ dnnl::memory::format_tag::nChw16c,     Shape(SizeVector{1, 1, 10, 10}) }, 1600},
            {{ dnnl::memory::format_tag::nhwc,        Shape(SizeVector{4, 2, 10, 7 }) }, 560},
            {{ dnnl::memory::format_tag::nchw,        Shape(SizeVector{4, 2, 10, 7 }) }, 560},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(SizeVector{1, 1, 10, 10}) }, 800},
            {{ dnnl::memory::format_tag::nChw16c,     Shape(ngraph::PartialShape{{16}, {6}, {-1, -1}, {-1, -1}}) }, Shape::UNDEFINED_DIM},
            {{ dnnl::memory::format_tag::nhwc,        Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}) }, 122880},
            {{ dnnl::memory::format_tag::nchw,        Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}) }, 122880},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}) }, Shape::UNDEFINED_DIM},
    };

    for (const auto &p : payload) {
        getMaxMemSize(p.first.second, p.first.first, p.second);
    }
}

TEST(MemDescTest, getCurrentMemSize) {
    auto getCurrentMemSize = [&] (const Shape& shape, const dnnl::memory::format_tag formatTag, const size_t result) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::u8, formatTag);
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());

        ASSERT_EQ(result, dnnl_blk_desc_ptr->getCurrentMemSize());
        ASSERT_EQ(result, cpu_blk_desc_ptr->getCurrentMemSize());

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, formatTag};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            ASSERT_LT(result, desc_extra_ptr->getCurrentMemSize());
        }
    };

    std::pair<std::pair<dnnl::memory::format_tag, Shape>, size_t> payload[] {
            {{ dnnl::memory::format_tag::nChw16c,     Shape(SizeVector{1, 1, 10, 10}) }, 1600},
            {{ dnnl::memory::format_tag::nhwc,        Shape(SizeVector{4, 2, 10, 7 }) }, 560},
            {{ dnnl::memory::format_tag::nchw,        Shape(SizeVector{4, 2, 10, 7 }) }, 560},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(SizeVector{1, 1, 10, 10}) }, 800},
            {{ dnnl::memory::format_tag::nChw16c,     Shape(ngraph::PartialShape{{16}, {6}, {-1, -1}, {-1, -1}}) }, Shape::UNDEFINED_DIM},
            {{ dnnl::memory::format_tag::nhwc,        Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}) }, Shape::UNDEFINED_DIM},
            {{ dnnl::memory::format_tag::nchw,        Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}) }, Shape::UNDEFINED_DIM},
            {{ dnnl::memory::format_tag::nChw8c,      Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}) }, Shape::UNDEFINED_DIM},
    };

    for (const auto &p : payload) {
        getCurrentMemSize(p.first.second, p.first.first, p.second);
    }
}

TEST(MemDescTest, serializeFormat) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});
    Shape staticShape(SizeVector{1, 1, 10, 10});

    auto serializeFormat = [] (dnnl::memory::format_tag fmt, const Shape& shape, std::string str) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr;

        // DnnlBlockedMemoryDesc
        dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::u8, fmt);

        ASSERT_EQ(dnnl_blk_desc_ptr->serializeFormat(), str);

        // CpuBlockedMemoryDesc
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());

        ASSERT_EQ(cpu_blk_desc_ptr->serializeFormat(), str);
        ASSERT_EQ(cpu_blk_desc_ptr->serializeFormat(), dnnl_blk_desc_ptr->serializeFormat());

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            ASSERT_EQ(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->serializeFormat(), str);
        }
    };

    std::pair<std::pair<dnnl::memory::format_tag, Shape>, std::string> payload[] {
            {{ dnnl::memory::format_tag::nChw16c,     staticShape  }, "aBcd16b",    },  // auto blocked
            {{ dnnl::memory::format_tag::nhwc,        staticShape  }, "acdb",       },  // permuted
            {{ dnnl::memory::format_tag::nchw,        staticShape  }, "abcd",       },  // plain
            {{ dnnl::memory::format_tag::NChw16n16c,  staticShape  }, "ABcd16a16b", },  // blocked for 2 dims
            {{ dnnl::memory::format_tag::Acdb16a,     staticShape  }, "Acdb16a",    },  // same strides but not default order
            {{ dnnl::memory::format_tag::nChw16c,     dynamicShape }, "aBcd16b",    },  // auto blocked
            {{ dnnl::memory::format_tag::nhwc,        dynamicShape }, "acdb",       },  // permuted
            {{ dnnl::memory::format_tag::nchw,        dynamicShape }, "abcd",       },  // plain
            {{ dnnl::memory::format_tag::NChw16n16c,  dynamicShape }, "ABcd16a16b", },  // blocked for 2 dims
            {{ dnnl::memory::format_tag::Acdb16a,     dynamicShape }, "Acdb16a",    },  // same strides but not default order
    };

    for (const auto &p : payload) {
        serializeFormat(p.first.first, p.first.second, p.second);
    }
}

TEST(MemDescTest, redefineDesc) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});
    Shape shapeOrig(SizeVector{8, 16, 15, 17});
    Shape staticShape(SizeVector{1, 1, 10, 10});

    auto redefineDesc = [&] (dnnl::memory::format_tag fmt, const Shape& shape) {
        mkldnn::engine eng(dnnl::engine::kind::cpu, 0);
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_orig = std::make_shared<DnnlBlockedMemoryDesc>(shapeOrig, mkldnn::memory::data_type::u8,
                                                                                              dnnl::memory::format_tag::nhwc);

        // DnnlMemoryDesc
        dnnl::memory::desc orig_desc{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
        orig_desc.data.format_kind = dnnl_format_kind_undef;
        orig_desc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        MKLDNNMemory mkldnnMemDesc(eng);
        mkldnnMemDesc.Create(dnnl_blk_desc_orig);

        if (shape.isStatic()) {
            MemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc);
            mkldnnMemDesc.redefineDesc(desc_ptr);

            ASSERT_TRUE(mkldnnMemDesc.getDesc().isCompatible(*desc_ptr));
            ASSERT_FALSE(mkldnnMemDesc.getDesc().isCompatible(*dnnl_blk_desc_orig));
        } else {
            ASSERT_ANY_THROW(MemoryDescPtr desc_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc));
        }

        // DnnlBlockedMemoryDesc
        MKLDNNMemory mkldnnMemBlkDesc(eng);
        mkldnnMemBlkDesc.Create(dnnl_blk_desc_orig);
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr;
        dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::u8, dnnl::memory::format_tag::nhwc);
        mkldnnMemBlkDesc.redefineDesc(dnnl_blk_desc_ptr);

        ASSERT_TRUE(mkldnnMemBlkDesc.getDesc().isCompatible(*dnnl_blk_desc_ptr));
        ASSERT_FALSE(mkldnnMemBlkDesc.getDesc().isCompatible(*dnnl_blk_desc_orig));

        // CpuBlockedMemoryDesc
        MKLDNNMemory mkldnnMemCpuBlkDesc(eng);
        mkldnnMemCpuBlkDesc.Create(dnnl_blk_desc_orig);
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());
        mkldnnMemCpuBlkDesc.redefineDesc(cpu_blk_desc_ptr);

        ASSERT_TRUE(mkldnnMemCpuBlkDesc.getDesc().isCompatible(*cpu_blk_desc_ptr));
        ASSERT_FALSE(mkldnnMemCpuBlkDesc.getDesc().isCompatible(*dnnl_blk_desc_orig));

        // DnnlBlockedMemoryDesc with extra data
        // TODO [DS]: Should be added case for shape.isDynamic
        if (shape.isStatic()) {
            dnnl::memory::desc orig_desc_extra{MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims()), dnnl::memory::data_type::u8, fmt};
            orig_desc_extra.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
            orig_desc_extra.data.extra.compensation_mask = 1;
            orig_desc_extra.data.extra.scale_adjust = 2.0f;
            MemoryDescPtr desc_extra_ptr = MKLDNNExtensionUtils::makeDescriptor(orig_desc_extra);
            ASSERT_TRUE(desc_extra_ptr->as<DnnlBlockedMemoryDesc>()->getDnnlDesc() == orig_desc_extra);
            mkldnnMemBlkDesc.Create(dnnl_blk_desc_orig);
            mkldnnMemBlkDesc.redefineDesc(desc_extra_ptr);
            ASSERT_TRUE(mkldnnMemBlkDesc.getDesc().isCompatible(*desc_extra_ptr));
            ASSERT_FALSE(mkldnnMemBlkDesc.getDesc().isCompatible(*dnnl_blk_desc_orig));
        }
    };

    std::pair<dnnl::memory::format_tag, Shape> payload[] {
            { dnnl::memory::format_tag::nChw16c,     staticShape },  // auto blocked
            { dnnl::memory::format_tag::nhwc,        staticShape },  // permuted
            { dnnl::memory::format_tag::nchw,        staticShape },  // plain
            { dnnl::memory::format_tag::NChw16n16c,  staticShape },  // blocked for 2 dims
            { dnnl::memory::format_tag::Acdb16a,     staticShape },  // same strides but not default order
            { dnnl::memory::format_tag::nChw16c,     dynamicShape }, // auto blocked
            { dnnl::memory::format_tag::nhwc,        dynamicShape }, // permuted
            { dnnl::memory::format_tag::nchw,        dynamicShape }, // plain
            { dnnl::memory::format_tag::NChw16n16c,  dynamicShape }, // blocked for 2 dims
            { dnnl::memory::format_tag::Acdb16a,     dynamicShape }, // same strides but not default order
    };

    for (const auto &p : payload) {
        redefineDesc(p.first, p.second);
    }
}

TEST(MemDescTest, isBlockedCCheck) {
    const auto dims = dnnl::memory::dims {3, 2, 5, 7};
    const auto type = dnnl::memory::data_type::u8;

    dnnl::memory::desc blck8_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};
    dnnl::memory::desc blck8_permCD_tdesc {dims, type, dnnl::memory::format_tag::aBdc16b};

    const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
    const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};

    dnnl::memory::desc blck8_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
    dnnl::memory::desc blck8_permCD_crop_tdesc = blck8_permCD_tdesc.submemory_desc(crop_dims, crop_off);

    ASSERT_TRUE(MKLDNNExtensionUtils::makeDescriptor(blck8_crop_tdesc)->hasLayoutType(LayoutType::nCsp8c));
    ASSERT_FALSE(MKLDNNExtensionUtils::makeDescriptor(blck8_permCD_crop_tdesc)->hasLayoutType(LayoutType::nCsp8c));
}

<<<<<<< HEAD
TEST(MakeUndefinedDnnlDesc, wrongType) {
    GTEST_SKIP();
}

TEST(MakeUndefinedDnnlDesc, checkRank) {
    using mkldnn::memory;
    const memory::data_type dataType = memory::data_type::u8;
    const memory::desc origin({10, 20, 15, 7}, dataType, memory::format_tag::nChw16c);

    MKLDNNPlugin::Shape pluginShapeWrongRank(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}});
    ASSERT_THROW(MKLDNNExtensionUtils::makeUndefinedDesc(origin, pluginShapeWrongRank), InferenceEngine::ParameterMismatch);

    MKLDNNPlugin::Shape pluginShapeRightRank(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}});
    MemoryDescPtr memDesc;
    ASSERT_NO_THROW(memDesc = MKLDNNExtensionUtils::makeUndefinedDesc(origin, pluginShapeRightRank));
    ASSERT_FALSE(memDesc->isDefined());
}

TEST(MakeUndefinedDnnlDesc, checkDims) {
    using mkldnn::memory;
    const memory::data_type dataType = memory::data_type::u8;
    const memory::desc origin({10, 20, 15, 7}, dataType, memory::format_tag::nChw16c);

    ngraph::PartialShape fullyUndef({{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}});
    for (size_t i = 0; i < fullyUndef.size(); ++i) {
        auto partialShape = fullyUndef;
        partialShape[i] = {3}; // just a number which is not equal to any origin dims
        ASSERT_THROW(MKLDNNExtensionUtils::makeUndefinedDesc(origin, MKLDNNPlugin::Shape(partialShape)), InferenceEngine::ParameterMismatch);
    }
    for (size_t i = 0; i < origin.dims().size(); ++i) {
        auto partialShape = fullyUndef;
        partialShape[i] = {origin.dims()[i]};
        MemoryDescPtr memDesc;
        ASSERT_NO_THROW(memDesc = MKLDNNExtensionUtils::makeUndefinedDesc(origin, MKLDNNPlugin::Shape(fullyUndef)));
        ASSERT_FALSE(memDesc->isDefined());
    }
}

TEST(MakeUndefinedDnnlDesc, checkLayout) {
    using mkldnn::memory;
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

    ngraph::PartialShape fullyUndef({{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}});

    for (const auto& item : payload) {
        dnnl::memory::format_tag fmt;
        dnnl::memory::dims dims;
        std::string strFormat;
        std::tie(fmt, dims, strFormat) = item;
        const memory::desc origin(dims, dataType, fmt);

        auto undefDesc = MKLDNNExtensionUtils::makeUndefinedDesc(origin, MKLDNNPlugin::Shape(fullyUndef));
        ASSERT_FALSE(undefDesc->isDefined());
        MKLDNNPlugin::DnnlBlockedMemoryDesc referenceDesc(MKLDNNPlugin::Shape(fullyUndef), dataType, fmt);
        ASSERT_TRUE(undefDesc->isCompatible(referenceDesc));
        ASSERT_EQ(undefDesc->serializeFormat(), strFormat);
        auto defDesc = undefDesc->cloneWithNewDims(MKLDNNExtensionUtils::convertToVectorDims(dims));
        ASSERT_TRUE(defDesc->isDefined());
        ASSERT_EQ(origin, defDesc->as<DnnlBlockedMemoryDesc>()->getDnnlDesc());
    }
}

TEST(MakeUndefinedDnnlDesc, extraData) {
    using mkldnn::memory;
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

    ngraph::PartialShape fullyUndef({{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}});

    for (const auto& item : payload) {
        dnnl::memory::format_tag fmt;
        dnnl::memory::dims dims;
        std::tie(fmt, dims) = item;
        memory::desc origin(dims, dataType, fmt);

        origin.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        origin.data.extra.compensation_mask = 1;
        origin.data.extra.scale_adjust = 2.0f;

        auto undefDesc = MKLDNNExtensionUtils::makeUndefinedDesc(origin, MKLDNNPlugin::Shape(fullyUndef));
        ASSERT_FALSE(undefDesc->isDefined());
        auto defDesc = undefDesc->cloneWithNewDims(MKLDNNExtensionUtils::convertToVectorDims(dims));
        ASSERT_TRUE(defDesc->isDefined());
        auto referenceDesc = MKLDNNExtensionUtils::makeDescriptor(origin);
        ASSERT_TRUE(defDesc->isCompatible(*referenceDesc));
        ASSERT_EQ(origin, defDesc->as<DnnlBlockedMemoryDesc>()->getDnnlDesc());
    }
}

=======
TEST(MemDescTest, getOrder) {
    Shape dynamicShape(ngraph::PartialShape{{16}, {7, 15}, {-1, -1}, {-1, -1}});
>>>>>>> e7039866a... Dynamic shapes test coverage

    auto getOrder = [] (const dnnl::memory::format_tag fmt, const Shape& shape, const VectorDims expected) {
        DnnlBlockedMemoryDescPtr dnnl_blk_desc_ptr = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::f32, fmt);
        CpuBlockedMemoryDescPtr cpu_blk_desc_ptr = std::make_shared<CpuBlockedMemoryDesc>(dnnl_blk_desc_ptr->getPrecision(), dnnl_blk_desc_ptr->getShape(),
                                                                                          dnnl_blk_desc_ptr->getBlockDims(), dnnl_blk_desc_ptr->getOrder(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPadding(),
                                                                                          dnnl_blk_desc_ptr->getOffsetPaddingToData(),
                                                                                          dnnl_blk_desc_ptr->getStrides());
    };

    std::pair<std::pair<dnnl::memory::format_tag, Shape>, VectorDims> payload[] {
            {{ dnnl::memory::format_tag::nChw16c,     Shape(SizeVector {1, 1, 10, 10}) }, SizeVector{0, 1, 2, 3, 1}   },  // auto blocked
            {{ dnnl::memory::format_tag::nhwc,        Shape(SizeVector {4, 2, 10, 7 }) }, SizeVector{0, 2, 3, 1}      },  // permuted
            {{ dnnl::memory::format_tag::nchw,        Shape(SizeVector {4, 2, 10, 7 }) }, SizeVector{0, 1, 2, 3}      },  // plain
            {{ dnnl::memory::format_tag::NChw16n16c,  Shape(SizeVector {4, 2, 10, 7 }) }, SizeVector{0, 1, 2, 3, 0, 1}},  // blocked for 2 dims
            {{ dnnl::memory::format_tag::BAcd16a16b,  Shape(SizeVector {4, 2, 10, 7 }) }, SizeVector{1, 0, 2, 3, 0, 1}},  // blocked and permuted outer dims
            {{ dnnl::memory::format_tag::Acdb16a,     Shape(SizeVector {96, 1, 7, 7 }) }, SizeVector{0, 2, 3, 1, 0}   },  // same strides but not default order
            {{ dnnl::memory::format_tag::nChw16c,                         dynamicShape }, SizeVector{0, 1, 2, 3, 1}   },  // auto blocked
            {{ dnnl::memory::format_tag::nhwc,                            dynamicShape }, SizeVector{0, 2, 3, 1}      },  // permuted
            {{ dnnl::memory::format_tag::nchw,                            dynamicShape }, SizeVector{0, 1, 2, 3}      },  // plain
            {{ dnnl::memory::format_tag::NChw16n16c,                      dynamicShape }, SizeVector{0, 1, 2, 3, 0, 1}},  // blocked for 2 dims
            {{ dnnl::memory::format_tag::BAcd16a16b,                      dynamicShape }, SizeVector{1, 0, 2, 3, 0, 1}},  // blocked and permuted outer dims
            {{ dnnl::memory::format_tag::Acdb16a,                         dynamicShape }, SizeVector{0, 2, 3, 1, 0}   },  // same strides but not default order
    };

    for (auto& p : payload) {
        getOrder(p.first.first, p.first.second, p.second);
    }
}

TEST(MemDescTest, UndefinedAndDefaultParams) {
    dnnl::memory::format_tag testCases[] {
        dnnl::memory::format_tag::nchw,
        dnnl::memory::format_tag::nhwc,
        dnnl::memory::format_tag::nChw8c,
        dnnl::memory::format_tag::nChw16c
    };

    // DnnlBlockedMemoryDesc with extra
    auto cloneWithParamsChangeDnnl = [](dnnl::memory::format_tag fmt) {
        dnnl::memory::desc refOneDnnDesc(dnnl::memory::dims{2, 3, 4, 5}, mkldnn::memory::data_type::u8, fmt);
        refOneDnnDesc.data.extra.flags = dnnl_memory_extra_flag_compensation_conv_s8s8;
        refOneDnnDesc.data.extra.compensation_mask = 1;
        refOneDnnDesc.data.extra.scale_adjust = 2.0f;
        auto refDesc = MKLDNNExtensionUtils::makeDescriptor(refOneDnnDesc);
        auto refDnnlBlkDesc = refDesc->as<DnnlBlockedMemoryDesc>();

        auto undefDesc = refDnnlBlkDesc->cloneWithUndefStridesAndOffset();
        auto undefDnnlBlkDesc = undefDesc->as<DnnlBlockedMemoryDesc>();
        ASSERT_EQ(refDnnlBlkDesc->getBlockDims(), undefDnnlBlkDesc->getBlockDims());
        ASSERT_EQ(refDnnlBlkDesc->getOrder(), undefDnnlBlkDesc->getOrder());
        ASSERT_EQ(refDnnlBlkDesc->getOffsetPaddingToData(), undefDnnlBlkDesc->getOffsetPaddingToData());
        // undef
        ASSERT_EQ(Shape::UNDEFINED_DIM, undefDnnlBlkDesc->getOffsetPadding());
        auto undefStrides = refDnnlBlkDesc->getStrides();
        std::fill(undefStrides.begin(), undefStrides.begin() + refDnnlBlkDesc->getShape().getRank(), Shape::UNDEFINED_DIM);
        ASSERT_EQ(undefStrides, undefDnnlBlkDesc->getStrides());
        ASSERT_FALSE(undefDnnlBlkDesc->isDefined());

        auto definedDesc = undefDnnlBlkDesc->cloneWithDefaultStridesAndOffset();
        auto definedDnnlBlkDesc = definedDesc->as<DnnlBlockedMemoryDesc>();
        ASSERT_TRUE(refOneDnnDesc == definedDnnlBlkDesc->as<DnnlMemoryDesc>()->getDnnlDesc());
        ASSERT_EQ(refDnnlBlkDesc->getBlockDims(), definedDnnlBlkDesc->getBlockDims());
        ASSERT_EQ(refDnnlBlkDesc->getOrder(), definedDnnlBlkDesc->getOrder());
        ASSERT_EQ(refDnnlBlkDesc->getOffsetPaddingToData(), definedDnnlBlkDesc->getOffsetPaddingToData());
        ASSERT_EQ(refDnnlBlkDesc->getOffsetPadding(), definedDnnlBlkDesc->getOffsetPadding());
        ASSERT_EQ(refDnnlBlkDesc->getStrides(), definedDnnlBlkDesc->getStrides());
        ASSERT_TRUE(refDnnlBlkDesc->isDefined());
    };

    for (const auto &tc : testCases) {
        cloneWithParamsChangeDnnl(tc);
    }

    // CpuBlockedMemoryDesc
    auto cloneWithParamsChangeCpu = [](dnnl::memory::format_tag fmt) {
        dnnl::memory::desc refOneDnnDesc(dnnl::memory::dims{2, 3, 4, 5}, mkldnn::memory::data_type::u8, fmt);
        auto refDesc = MemoryDescUtils::convertToBlockedMemoryDesc(MKLDNNExtensionUtils::makeDescriptor(refOneDnnDesc));

        auto undefDesc = refDesc->cloneWithUndefStridesAndOffset();
        auto undefCpuBlkDesc = undefDesc->as<BlockedMemoryDesc>();
        ASSERT_EQ(refDesc->getBlockDims(), undefCpuBlkDesc->getBlockDims());
        ASSERT_EQ(refDesc->getOrder(), undefCpuBlkDesc->getOrder());
        ASSERT_EQ(refDesc->getOffsetPaddingToData(), undefCpuBlkDesc->getOffsetPaddingToData());
        // undef
        ASSERT_EQ(Shape::UNDEFINED_DIM, undefCpuBlkDesc->getOffsetPadding());
        auto undefStrides = refDesc->getStrides();
        std::fill(undefStrides.begin(), undefStrides.begin() + refDesc->getShape().getRank(), Shape::UNDEFINED_DIM);
        ASSERT_EQ(undefStrides, undefCpuBlkDesc->getStrides());
        ASSERT_FALSE(undefCpuBlkDesc->isDefined());

        auto definedDesc = undefCpuBlkDesc->cloneWithDefaultStridesAndOffset();
        auto definedDnnlBlkDesc = definedDesc->as<BlockedMemoryDesc>();
        ASSERT_EQ(refDesc->getBlockDims(), definedDnnlBlkDesc->getBlockDims());
        ASSERT_EQ(refDesc->getOrder(), definedDnnlBlkDesc->getOrder());
        ASSERT_EQ(refDesc->getOffsetPaddingToData(), definedDnnlBlkDesc->getOffsetPaddingToData());
        ASSERT_EQ(refDesc->getOffsetPadding(), definedDnnlBlkDesc->getOffsetPadding());
        ASSERT_EQ(refDesc->getStrides(), definedDnnlBlkDesc->getStrides());
        ASSERT_TRUE(definedDnnlBlkDesc->isDefined());
    };

    for (const auto &tc : testCases) {
        cloneWithParamsChangeCpu(tc);
    }
}

TEST(makeDummyDesc, LowerBoundMoreThanDummyValue) {
    Shape shape(ngraph::PartialShape{1, 3, 85, {144, 1444}});
    auto desc = std::make_shared<DnnlBlockedMemoryDesc>(shape, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::nchw);
    ASSERT_FALSE(desc->isDefined());

    MemoryDescPtr definedDesc;
    ASSERT_NO_THROW(definedDesc = MemoryDescUtils::makeDummyDesc(*desc));

    ASSERT_TRUE(definedDesc->isDefined());
    ASSERT_EQ((VectorDims{1, 3, 85, 144}), definedDesc->getShape().getStaticDims());
}
