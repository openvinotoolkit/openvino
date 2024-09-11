// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-spec-builders.h>
#include <gmock/gmock.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <openvino/core/shape.hpp>
#include <openvino/core/strides.hpp>
#include <openvino/core/type/element_type.hpp>
#include "memory_desc/blocked_memory_desc.h"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"

#include "cpu_memory.h"
#include "cpu_tensor.h"
#include "openvino/runtime/itensor.hpp"
#include "common_test_utils/test_assertions.hpp"


using namespace ov::intel_cpu;

using CPUTensorTest = ::testing::Test;

class MockBlockedMemoryDesc : public BlockedMemoryDesc {
public:
    MockBlockedMemoryDesc(const Shape& _shape) : MemoryDesc(_shape, Blocked) {}

    MOCK_METHOD(ov::element::Type, getPrecision, (), (const, override));
    MOCK_METHOD(MemoryDescPtr, clone, (), (const, override));
    MOCK_METHOD(size_t, getOffsetPadding, (), (const, override));

    MOCK_METHOD(MemoryDescPtr, cloneWithNewDimsImp, (const VectorDims&), (const, override));

    MOCK_METHOD(MemoryDescPtr, cloneWithNewPrecision, (const ov::element::Type), (const, override));
    MOCK_METHOD(bool, isCompatible, (const MemoryDesc&), (const, override));

    MOCK_METHOD(bool, hasLayoutType, (LayoutType), (const, override));

    MOCK_METHOD(size_t, getMaxMemSize, (), (const, override));

    MOCK_METHOD(const VectorDims&, getBlockDims, (), (const, override));
    MOCK_METHOD(const VectorDims&, getOrder, (), (const, override));
    MOCK_METHOD(const VectorDims&, getOffsetPaddingToData, (), (const, override));
    MOCK_METHOD(const VectorDims&, getStrides, (), (const, override));
    MOCK_METHOD(bool, blocksExtended, (), (const, override));
    MOCK_METHOD(size_t, getPaddedElementsCount, (), (const, override));
    MOCK_METHOD(bool, isCompatible, (const BlockedMemoryDesc &, CmpMask), (const, override));

    MOCK_METHOD(void, setPrecision, (ov::element::Type), (override));

    MOCK_METHOD(size_t, getCurrentMemSizeImp, (), (const, override));

    MOCK_METHOD(size_t, getElementOffset, (size_t), (const, override));
    MOCK_METHOD(bool, canComputeMemSizeZeroDims, (), (const, override));
    MOCK_METHOD(bool, isDefinedImp, (), (const, override));
};

class MockIMemory : public IMemory {
public:
    MockIMemory(MemoryDescPtr desc) : m_pMemDesc(desc) {}
    MockIMemory(const MemoryDesc& desc) : m_pMemDesc(desc.clone()) {}

    MOCK_METHOD(MemoryDesc&, getDesc, (), (const, override));
    MOCK_METHOD(MemoryDescPtr, getDescPtr, (), (const, override));

    MOCK_METHOD(size_t, getSize, (), (const, override));
    MOCK_METHOD(const Shape&, getShape, (), (const, override));
    MOCK_METHOD(const VectorDims&, getStaticDims, (), (const, override));

    MOCK_METHOD(void, redefineDesc, (MemoryDescPtr), (override));
    MOCK_METHOD(void, load, (const IMemory&, bool), (const, override));
    MOCK_METHOD(MemoryBlockPtr, getMemoryBlock, (), (const, override));

    MOCK_METHOD(dnnl::memory, getPrimitive, (), (const, override));
    MOCK_METHOD(void, nullify, (), (override));
    MOCK_METHOD(void*, getData, (), (const, override));

    void set_memDesc(MemoryDescPtr memdesc) { m_pMemDesc = memdesc; }
    void set_memDesc(const MemoryDesc& memdesc) { m_pMemDesc = memdesc.clone(); }
    MemoryDesc& get_memDesc() const { return *m_pMemDesc; }
    MemoryDescPtr get_memDescPtr() { return m_pMemDesc; }

private:
    MemoryDescPtr m_pMemDesc;
};

// helper to get byte strides from strides.
static ov::Strides byte_strides(const ov::Strides& strides, const ov::element::Type& type) {
    ov::Strides byte_strides(strides.size());
    for (size_t i = 0; i < strides.size(); ++i)
        byte_strides[i] = strides[i] * type.size();
    return byte_strides;
}

// helper to create Memory of ncsp layout.
inline MemoryDescPtr create_memdesc(ov::element::Type prec, const Shape& shape, const VectorDims& strides = {}) {
    ov::Shape ov_shape = shape.toPartialShape().to_shape();
    const std::size_t totalSize = ov::shape_size(ov_shape);
    auto elem_type = prec;

    auto memdesc = std::make_shared<MockBlockedMemoryDesc>(shape);
    ::testing::Mock::AllowLeak(memdesc.get());

    EXPECT_CALL(*memdesc, hasLayoutType(::testing::Eq(LayoutType::ncsp))).WillRepeatedly(::testing::Return(true));

    EXPECT_CALL(*memdesc, getPrecision).WillRepeatedly(::testing::Return(prec));
    EXPECT_CALL(*memdesc, getStrides).WillRepeatedly(::testing::ReturnRef(strides));

    EXPECT_CALL(*memdesc, canComputeMemSizeZeroDims).WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*memdesc, isDefinedImp).WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*memdesc, getCurrentMemSizeImp).WillRepeatedly(::testing::Return(totalSize * elem_type.size()));

    return memdesc;
}

inline MemoryPtr create_memory(MemoryDescPtr memdesc) {
    auto memptr = std::make_shared<MockIMemory>(memdesc);
    ::testing::Mock::AllowLeak(memptr.get());

    // getDesc
    EXPECT_CALL(*memptr, getDescPtr)
        .Times(::testing::AnyNumber())
        .WillRepeatedly([memptr]() {
                        return memptr->get_memDescPtr();
                    });
    EXPECT_CALL(*memptr, getDesc).WillRepeatedly(::testing::ReturnRef(memptr->get_memDesc()));

    // data
    static size_t memSize = 0;
    EXPECT_CALL(*memptr, getData)
        .WillRepeatedly([memptr]() {
                        auto memdesc = memptr->get_memDescPtr();
                        auto required = memdesc->getCurrentMemSize();
                        if (memSize >= required) {
                            return reinterpret_cast<void*>(memSize);
                        } else {
                            memSize = required;
                            return reinterpret_cast<void*>(required);
                        }
                    });

    // redefineDesc
    ON_CALL(*memptr, redefineDesc).WillByDefault([memptr](MemoryDescPtr desc) {
                memptr->set_memDesc(desc);
            });
    EXPECT_CALL(*memptr, redefineDesc).Times(::testing::AtLeast(1));

    return memptr;
}

TEST_F(CPUTensorTest, canCreateTensor) {
    Shape shape{4, 3, 2};
    ov::Shape ov_shape = shape.toPartialShape().to_shape();
    auto strides = ov::Strides({6, 2, 1});
    const std::size_t totalSize = ov::shape_size(ov_shape);
    ov::element::Type elem_type = ov::element::f32;

    auto memptr = create_memory(create_memdesc(ov::element::f32, shape, strides));
    {
        std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(memptr);
        ASSERT_EQ(totalSize, t->get_size());
        ASSERT_NE(nullptr, t->data());
        ASSERT_EQ(elem_type, t->get_element_type());
        ASSERT_EQ(ov_shape, t->get_shape());
        ASSERT_NE(ov_shape, t->get_strides());
        ASSERT_EQ(byte_strides(ov::Strides({6, 2, 1}), t->get_element_type()), t->get_strides());
        ASSERT_EQ(elem_type.size() * totalSize, t->get_byte_size());
        ASSERT_THROW(t->data(ov::element::i64), ov::Exception);
        ASSERT_THROW(t->data<std::int32_t>(), ov::Exception);
    }
}

TEST_F(CPUTensorTest, canAccessF16Tensor) {
    Shape shape = {4, 3, 2};
    auto strides = ov::Strides({6, 2, 1});

    auto memptr = create_memory(create_memdesc(ov::element::f16, shape, strides));
    {
        std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(memptr);
        EXPECT_NE(nullptr, t->data());
        ASSERT_EQ(ov::element::f16, t->get_element_type());
        EXPECT_NO_THROW(t->data(ov::element::f16));
        EXPECT_NO_THROW(t->data<ov::float16>());
        EXPECT_THROW(t->data<ov::bfloat16>(), ov::Exception);
        EXPECT_THROW(t->data<std::uint16_t>(), ov::Exception);
        EXPECT_THROW(t->data<std::int16_t>(), ov::Exception);
    }
}

// SetShape
TEST_F(CPUTensorTest, canSetShape) {
    const Shape origShape = {1, 2, 3};
    const ov::Shape ov_origShape = origShape.toPartialShape().to_shape();
    auto strides = ov::Strides({6, 3, 1});
    auto memdesc = create_memdesc(ov::element::f32, origShape, strides);
    auto memptr = create_memory(memdesc);
    std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(memptr);

    const Shape newShape({4, 5, 6});
    const ov::Shape ov_newShape = newShape.toPartialShape().to_shape();
    auto new_strides = ov::Strides{30, 6, 1};
    auto new_memdesc = create_memdesc(ov::element::f32, newShape, new_strides);

    // set_shape to a bigger memory
    {
        auto blocked_memdesc = dynamic_cast<MockBlockedMemoryDesc*>(memdesc.get());
        EXPECT_CALL(*blocked_memdesc, cloneWithNewDimsImp).WillRepeatedly(::testing::Return(new_memdesc));

        const void* orig_data = t->data();
        ASSERT_EQ(t->get_shape(), ov_origShape);
        OV_ASSERT_NO_THROW(t->set_shape(ov_newShape));
        ASSERT_EQ(ov_newShape, t->get_shape());
        ASSERT_EQ(byte_strides(ov::row_major_strides(ov_newShape), t->get_element_type()), t->get_strides());
        ASSERT_NE(orig_data, t->data());
    }

    // set_shape for smaller memory - does not perform reallocation
    {
        auto new_blocked_memdesc = dynamic_cast<MockBlockedMemoryDesc*>(new_memdesc.get());
        EXPECT_CALL(*new_blocked_memdesc, cloneWithNewDimsImp).WillRepeatedly(::testing::Return(memdesc));
        const void* orig_data = t->data();
        t->set_shape(ov_origShape);
        ASSERT_EQ(ov_origShape, t->get_shape());
        ASSERT_EQ(orig_data, t->data());
    }
}

TEST_F(CPUTensorTest, canSyncMemoryAndTensor) {
    const Shape origShape = {1, 2, 3};
    const ov::Shape ov_origShape = origShape.toPartialShape().to_shape();
    auto strides = ov::Strides({6, 3, 1});
    auto memdesc = create_memdesc(ov::element::f32, origShape, strides);
    auto memptr = create_memory(memdesc);
    std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(memptr);

    ASSERT_EQ(memptr->getDescPtr()->getShape().toPartialShape().to_shape(), t->get_shape());
    ASSERT_EQ(byte_strides(memptr->getDescWithType<BlockedMemoryDesc>()->getStrides(), t->get_element_type()), t->get_strides());

    const Shape newShape({4, 5, 6});
    const ov::Shape ov_newShape = newShape.toPartialShape().to_shape();
    auto new_strides = ov::Strides{30, 6, 1};
    auto new_memdesc = create_memdesc(ov::element::f32, newShape, new_strides);

    // reallocate memory out boundary of tensor instance
    {
        auto blocked_memdesc = dynamic_cast<MockBlockedMemoryDesc*>(memdesc.get());
        EXPECT_CALL(*blocked_memdesc, cloneWithNewDimsImp).WillRepeatedly(::testing::Return(new_memdesc));

        auto desc2 = memptr->getDescPtr()->cloneWithNewDims(newShape.getStaticDims(), true);
        memptr->redefineDesc(desc2);
        ASSERT_EQ(memptr->getDescPtr()->getShape().toPartialShape().to_shape(), t->get_shape());
        ASSERT_EQ(byte_strides(memptr->getDescWithType<BlockedMemoryDesc>()->getStrides(), t->get_element_type()), t->get_strides());
    }
}
