// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpu_memory.h>
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/blocked_desc_creator.h"
#include <dnnl_extension_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace ov::intel_cpu;
using namespace testing;

/* ======================================= BASE ZERO DIM TEST ======================================= */
class MemDescWithZeroDimsBaseTest: public ::testing::Test {
protected:
    Shape shape;
    dnnl::memory::format_tag fmt;
    const ov::element::Type precision = ov::element::f32;

    void validate(const BlockedMemoryDesc& desc, const VectorDims& expectedStrieds, size_t offsetSize, size_t offsetPaddingSize,
                  size_t maxMemSize, bool orderCheckSkip = false) {
        VectorDims expectedBlkDims;
        VectorDims expectedOrder;
        {
            auto origShape = shape.toPartialShape();
            auto replaceShape = origShape;
            std::replace(replaceShape.begin(), replaceShape.end(), ov::Dimension(0), ov::Dimension(3));
            Shape dummyShape(replaceShape);
            DnnlBlockedMemoryDesc dummyDesc(dummyShape, DnnlExtensionUtils::ElementTypeToDataType(precision), fmt);
            expectedBlkDims = dummyDesc.getBlockDims();
            expectedOrder = dummyDesc.getOrder();
            for (size_t i = 0; i < dummyShape.getRank(); i++) {
                if (origShape[expectedOrder[i]] == ov::Dimension(0)) {
                    expectedBlkDims[i] = 0;
                }
            }
        }

        ASSERT_EQ(shape.getDims(), desc.getShape().getDims());
        ASSERT_EQ(shape.getMinDims(), desc.getShape().getMinDims());
        ASSERT_EQ(shape.getMaxDims(), desc.getShape().getMaxDims());

        ASSERT_EQ(expectedStrieds, desc.getStrides());
        ASSERT_EQ(expectedBlkDims, desc.getBlockDims());
        if (!orderCheckSkip) {
            ASSERT_EQ(expectedOrder, desc.getOrder());
        }

        ASSERT_EQ(0, desc.getPaddedElementsCount());
        ASSERT_EQ(maxMemSize, desc.getMaxMemSize());
        ASSERT_EQ(maxMemSize, desc.getCurrentMemSize());

        ASSERT_EQ(offsetSize, desc.getOffsetPadding());
        ASSERT_EQ(VectorDims(expectedBlkDims.size(), offsetPaddingSize), desc.getOffsetPaddingToData());
    }

    virtual std::pair<DnnlBlockedMemoryDesc, CpuBlockedMemoryDesc> createDescs() const {
        DnnlBlockedMemoryDesc descDnnl(precision, shape);
        CpuBlockedMemoryDesc descCpu(precision, shape);
        return {descDnnl, descCpu};
    }

    void Run() {
        const size_t offset = 0, offsetPadding = 0;

        auto descs = createDescs();
        DnnlBlockedMemoryDesc descDnnl(descs.first);
        CpuBlockedMemoryDesc descCpu(descs.second);

        VectorDims zeroStrides(descDnnl.getBlockDims().size(), 0);
        validate(descDnnl, zeroStrides, offset, offsetPadding, 0);
        validate(descCpu, zeroStrides, offset, offsetPadding, 0);

        ASSERT_TRUE(descDnnl.isCompatible(descCpu));
        ASSERT_TRUE(descCpu.isCompatible(descDnnl));
    }
};

/* ======================================= TEST DATA ======================================= */
const std::vector<Shape> staticShapes = {
    Shape(VectorDims{0, 32, 48, 64}),
    Shape(VectorDims{16, 0, 48, 64}),
    Shape(VectorDims{16, 32, 0, 64}),
    Shape(VectorDims{16, 32, 48, 0}),
    Shape(VectorDims{16, 32, 0, 0}),
    Shape(VectorDims{0, 0, 48, 64}),
    Shape(VectorDims{16, 0, 0, 64}),
    Shape(VectorDims{0, 0, 0, 64}),
    Shape(VectorDims{16, 0, 0, 0}),
    Shape(VectorDims{0, 0, 0, 0})
};

const std::vector<Shape> dynamicShapes = {
    Shape(ov::PartialShape{0, -1, {0, 48}, -1}),
    Shape(ov::PartialShape{16, 0, -1, {0, 64}}),
    Shape(ov::PartialShape{-1, -1, 0, -1}),
    Shape(ov::PartialShape{{0, 16}, -1, {0, 48}, 0}),
    Shape(ov::PartialShape{-1, 32, 0, 0}),
    Shape(ov::PartialShape{0, 0, 48, -1}),
    Shape(ov::PartialShape{{0, 16}, 0, 0, 64}),
    Shape(ov::PartialShape{0, 0, 0, -1}),
    Shape(ov::PartialShape{{0, 16}, 0, 0, 0}),
    Shape(ov::PartialShape{0, 0, 0, 0})
};

const std::vector<dnnl::memory::format_tag> fmts = {
    dnnl::memory::format_tag::nchw,
    dnnl::memory::format_tag::nhwc,
    dnnl::memory::format_tag::nChw8c,
    dnnl::memory::format_tag::nChw16c,
    dnnl::memory::format_tag::NChw16n16c,
    dnnl::memory::format_tag::Acdb16a
};

/* ======================================= SPECIFIC TEST CASES ======================================= */
using MemDescWithZeroDimsParams = std::tuple<dnnl::memory::format_tag,
                                             Shape>;

class MemDescWithZeroDimsFmtTest: public testing::WithParamInterface<MemDescWithZeroDimsParams>,
                                  public MemDescWithZeroDimsBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MemDescWithZeroDimsParams> &obj) {
        Shape shape;
        dnnl::memory::format_tag fmt;
        std::tie(fmt, shape) = obj.param;
        std::ostringstream result;
        result << "Shape=" << shape.toString();
        result << "_Fmt=" << dnnl::utils::fmt2str(fmt);
        return result.str();
    }

    std::pair<DnnlBlockedMemoryDesc, CpuBlockedMemoryDesc> createDescs() const override {
        DnnlBlockedMemoryDesc descDnnl(shape, DnnlExtensionUtils::ElementTypeToDataType(precision), fmt);
        CpuBlockedMemoryDesc descCpu(precision, shape, descDnnl.getBlockDims(), descDnnl.getOrder());
        return {descDnnl, descCpu};
    }

protected:
    void SetUp() override {
        std::tie(fmt, shape) = this->GetParam();
        ASSERT_TRUE(shape.hasZeroDims()) << "Can't run MemDescWithZeroDimsTest, because shape doesn't contain zero dims";
    }
};

TEST_P(MemDescWithZeroDimsFmtTest, CreateDescWithFmt) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MemDescWithZeroDimsFmtTest_static, MemDescWithZeroDimsFmtTest,
                         ::testing::Combine(::testing::ValuesIn(fmts),
                                            ::testing::ValuesIn(staticShapes)),
                         MemDescWithZeroDimsFmtTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MemDescWithZeroDimsFmtTest_dynamic, MemDescWithZeroDimsFmtTest,
                         ::testing::Combine(::testing::ValuesIn(fmts),
                                            ::testing::ValuesIn(dynamicShapes)),
                         MemDescWithZeroDimsFmtTest::getTestCaseName);

class MemDescWithZeroDimsPlanarTest: public testing::WithParamInterface<Shape>,
                                     public MemDescWithZeroDimsBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Shape> &obj) {
        Shape shape;
        shape = obj.param;
        std::ostringstream result;
        result << "Shape=" << shape.toString();
        return result.str();
    }
protected:
    void SetUp() override {
        shape = this->GetParam();
        fmt = dnnl::memory::format_tag::nchw;
        ASSERT_TRUE(shape.hasZeroDims()) << "Can't run MemDescWithZeroDimsTest, because shape doesn't contain zero dims";
    }
};

TEST_P(MemDescWithZeroDimsPlanarTest, CreateDescPlanar) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MemDescWithZeroDimsPlanarTest, MemDescWithZeroDimsPlanarTest,
                         ::testing::ValuesIn(staticShapes),
                         MemDescWithZeroDimsPlanarTest::getTestCaseName);

using MemDescWithZeroDimsCloneNewDimsParams = std::tuple<dnnl::memory::format_tag, // memory format
                                              Shape,                               // dynamic shapes
                                              Shape>;                              // static shapes

class MemDescWithZeroDimsCloneNewDimsTest: public testing::WithParamInterface<MemDescWithZeroDimsCloneNewDimsParams>,
                                           public MemDescWithZeroDimsBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MemDescWithZeroDimsCloneNewDimsParams> &obj) {
        Shape shapeDynamic, shapeClone;
        dnnl::memory::format_tag fmt;
        std::tie(fmt, shapeDynamic, shapeClone) = obj.param;
        std::ostringstream result;
        result << "ShapeDynamic=" << shapeDynamic.toString();
        result << "_ShapeClone=" << shapeClone.toString();
        result << "_Fmt=" << dnnl::utils::fmt2str(fmt);
        return result.str();
    }
protected:
    Shape shapeDynamic;

    void SetUp() override {
        std::tie(fmt, shapeDynamic, shape) = this->GetParam();
        ASSERT_TRUE(shape.hasZeroDims()) << "Can't run MemDescWithZeroDimsTest, because shape doesn't contain zero dims";
    }
};

TEST_P(MemDescWithZeroDimsCloneNewDimsTest, CloneWithNewDims) {
    DnnlBlockedMemoryDesc dynamicDescDnnl(shapeDynamic, DnnlExtensionUtils::ElementTypeToDataType(precision), fmt);
    CpuBlockedMemoryDesc dynamicDescCpu(precision, shape, dynamicDescDnnl.getBlockDims(), dynamicDescDnnl.getOrder());
    const size_t offset = 0, offsetPadding = 0;
    VectorDims zeroStrides(dynamicDescDnnl.getBlockDims().size(), 0);

    const auto clonedDescDnnl = dynamicDescDnnl.cloneWithNewDims(shape.getStaticDims());
    const auto clonedDescCpu = dynamicDescCpu.cloneWithNewDims(shape.getStaticDims());

    // can't compute order correct since strides equal
    const auto& dims = shape.getDims();
    bool skipOrderCheck = std::all_of(dims.begin() + 1, dims.end(), [](const size_t& dim) { return dim == 0; });
    validate(*clonedDescDnnl->as<BlockedMemoryDesc>(), zeroStrides, offset, offsetPadding, 0, skipOrderCheck);
    validate(*clonedDescCpu->as<BlockedMemoryDesc>(), zeroStrides, offset, offsetPadding, 0);
}

const std::vector<Shape> srcDynShapes = {
    Shape(ov::PartialShape({-1, -1, -1, -1})),
    Shape(ov::PartialShape({{0, 16}, {0, 32}, {0, 48}, {0, 64}}))
};

INSTANTIATE_TEST_SUITE_P(smoke_MemDescWithZeroDimsCloneNewDimsTest, MemDescWithZeroDimsCloneNewDimsTest,
                         ::testing::Combine(::testing::ValuesIn(fmts),
                                            ::testing::ValuesIn(srcDynShapes),
                                            ::testing::ValuesIn(staticShapes)),
                         MemDescWithZeroDimsCloneNewDimsTest::getTestCaseName);
