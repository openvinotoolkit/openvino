// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-spec-builders.h>
#include <gmock/gmock.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <openvino/core/shape.hpp>
#include <openvino/core/strides.hpp>
#include <openvino/core/type/element_type.hpp>
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"

#include "cpu_memory.h"
#include "cpu_tensor.h"
#include "openvino/runtime/itensor.hpp"
#include "common_test_utils/test_assertions.hpp"

using namespace ov::intel_cpu;

using CPUTensorExtTest = ::testing::Test;

static ov::Strides byteStrides(const ov::Strides& strides, const ov::element::Type& type) {
    ov::Strides byte_strides(strides.size());
    for (size_t i = 0; i < strides.size(); ++i)
        byte_strides[i] = strides[i] * type.size();
    return byte_strides;
}

inline MemoryPtr create_memory(ov::element::Type prc, const Shape& shape) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    CpuBlockedMemoryDescPtr desc;
    desc = std::make_shared<CpuBlockedMemoryDesc>(prc, shape);
    return std::make_shared<Memory>(eng, desc);
}

TEST_F(CPUTensorExtTest, canCreateTensor) {
    Shape shape{4, 3, 2};
    ov::Shape ov_shape = shape.toPartialShape().to_shape();

    std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(create_memory(ov::element::f32, shape));
    const std::size_t totalSize = ov::shape_size(ov_shape);
    ASSERT_EQ(totalSize, t->get_size());
    ASSERT_NE(nullptr, t->data());
    ASSERT_EQ(ov::element::f32, t->get_element_type());
    ASSERT_EQ(ov_shape, t->get_shape());
    ASSERT_NE(ov_shape, t->get_strides());
    ASSERT_EQ(byteStrides(ov::Strides({6, 2, 1}), t->get_element_type()), t->get_strides());
    ASSERT_EQ(ov::element::f32.size() * totalSize, t->get_byte_size());
    ASSERT_THROW(t->data(ov::element::i64), ov::Exception);
    ASSERT_THROW(t->data<std::int32_t>(), ov::Exception);
}

TEST_F(CPUTensorExtTest, canAccessF16Tensor) {
    Shape shape = {4, 3, 2};
    std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(create_memory(ov::element::f16, shape));
    EXPECT_NE(nullptr, t->data());
    ASSERT_EQ(ov::element::f16, t->get_element_type());
    EXPECT_NO_THROW(t->data(ov::element::f16));
    EXPECT_NO_THROW(t->data<ov::float16>());
    EXPECT_THROW(t->data<ov::bfloat16>(), ov::Exception);
    EXPECT_THROW(t->data<std::uint16_t>(), ov::Exception);
    EXPECT_THROW(t->data<std::int16_t>(), ov::Exception);
}

// SetShape
TEST_F(CPUTensorExtTest, canSetShape) {
    const ov::Shape origShape({1, 2, 3});
    std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(create_memory(ov::element::f32, {1, 2, 3}));
    const ov::Shape newShape({4, 5, 6});

    const void* orig_data = t->data();
    ASSERT_EQ(t->get_shape(), origShape);
    OV_ASSERT_NO_THROW(t->set_shape({4, 5, 6}));
    ASSERT_EQ(newShape, t->get_shape());
    ASSERT_EQ(byteStrides(ov::row_major_strides(newShape), t->get_element_type()), t->get_strides());
    ASSERT_NE(orig_data, t->data());

    // set_shape for smaller memory - does not perform reallocation
    {
        orig_data = t->data();
        t->set_shape(origShape);
        ASSERT_EQ(origShape, t->get_shape());
        ASSERT_EQ(orig_data, t->data());
    }
}

TEST_F(CPUTensorExtTest, emptySize) {
    ov::PartialShape pshape{0, 3, 2};
    Shape shape{pshape};
    const ov::Shape origShape({0, 3, 2});

    std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(create_memory(ov::element::f32, shape));

    ASSERT_EQ(ov::element::f32, t->get_element_type());
    ASSERT_EQ(0, t->get_size());
    ASSERT_EQ(0, t->get_byte_size());
    ASSERT_EQ(origShape, t->get_shape());
    ASSERT_EQ(byteStrides(ov::Strides({0, 0, 0}), t->get_element_type()), t->get_strides());
    EXPECT_NO_THROW(t->data());
}

TEST_F(CPUTensorExtTest, canCreateTensorWithDynamicShape) {
    ov::PartialShape pshape{-1, 3, 2};
    Shape shape{pshape};

    std::shared_ptr<ov::ITensor> t;

    // construct with memory with dynamic shape
    OV_ASSERT_NO_THROW(t = std::make_shared<ov::intel_cpu::Tensor>(create_memory(ov::element::f32, shape)));
    ASSERT_THROW(t->get_shape(), ov::Exception);
    ASSERT_THROW(t->get_strides(), ov::Exception);

    // change memory to dynamic shape
    {
        auto memptr = create_memory(ov::element::f32, {4, 3, 2});
        OV_ASSERT_NO_THROW(t = std::make_shared<ov::intel_cpu::Tensor>(memptr));

        ov::PartialShape pshape{{1, 10}, 3, 2};
        CpuBlockedMemoryDescPtr desc2 = std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape(pshape));
        memptr->redefineDesc(desc2);
        ASSERT_THROW(t->get_shape(), ov::Exception);
        ASSERT_THROW(t->get_strides(), ov::Exception);
    }

    // set_shape
    const ov::Shape newShape({4, 0, 2});
    OV_ASSERT_NO_THROW(t = std::make_shared<ov::intel_cpu::Tensor>(create_memory(ov::element::f32, {4, 3, 2})));

    const void* orig_data = t->data();
    OV_ASSERT_NO_THROW(t->set_shape({4, 0, 2}));
    ASSERT_EQ(newShape, t->get_shape());
    ASSERT_EQ(ov::Strides({0, 0, 0}), t->get_strides());
    ASSERT_EQ(orig_data, t->data());
}

TEST_F(CPUTensorExtTest, canSyncMemoryAndTensor) {
    Shape orig_shape{4, 3, 2};

    auto memptr = create_memory(ov::element::f32, orig_shape);
    std::shared_ptr<ov::ITensor> t = std::make_shared<ov::intel_cpu::Tensor>(memptr);
    ASSERT_EQ(memptr->getDescPtr()->getShape().toPartialShape().to_shape(), t->get_shape());
    ASSERT_EQ(byteStrides(memptr->getDescWithType<BlockedMemoryDesc>()->getStrides(), t->get_element_type()), t->get_strides());

    // reallocate memory out boundary of tensor instance
    {
        Shape new_shape{1, 5, 2};

        auto desc2 = memptr->getDescPtr()->cloneWithNewDims(new_shape.getStaticDims(), true);
        memptr->redefineDesc(desc2);
        ASSERT_EQ(memptr->getDescPtr()->getShape().toPartialShape().to_shape(), t->get_shape());
        ASSERT_EQ(byteStrides(memptr->getDescWithType<BlockedMemoryDesc>()->getStrides(), t->get_element_type()), t->get_strides());
    }
}
