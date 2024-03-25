// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "openvino/core/type/element_type.hpp"

using namespace ov::intel_cpu;

TEST(MemoryTest, EmptyMemoryDescVerifyPublicInterface) {
    const auto emptyDesc = MemoryDescUtils::makeEmptyDesc();
    ASSERT_EQ(emptyDesc->getType(), MemoryDescType::Empty);

    ASSERT_EQ(emptyDesc->getShape(), Shape{0});

    ASSERT_TRUE(emptyDesc->empty());

    ASSERT_TRUE(emptyDesc->clone()->empty());

    ASSERT_EQ(emptyDesc->getPrecision(), ov::element::undefined);

    ASSERT_EQ(emptyDesc->getOffsetPadding(), 0);

    for (const auto& layout : {LayoutType::ncsp, LayoutType::nspc, LayoutType::nCsp8c, LayoutType::nCsp16c}) {
        ASSERT_FALSE(emptyDesc->hasLayoutType(layout));
    }

    ASSERT_EQ(emptyDesc->serializeFormat(), "empty");

    ASSERT_EQ(emptyDesc->getMaxMemSize(), 0);

    ASSERT_THROW(emptyDesc->cloneWithNewPrecision(ov::element::f32), ov::Exception);

    // compatible with empty memory desc
    ASSERT_TRUE(emptyDesc->isCompatible(*emptyDesc->clone()));
    // not compatible with any other memory desc
    ASSERT_FALSE(emptyDesc->isCompatible(CpuBlockedMemoryDesc{ov::element::f32, Shape{1, 2, 3}}));
    ASSERT_FALSE(emptyDesc->isCompatible(DnnlBlockedMemoryDesc{ov::element::u8, Shape{1}}));
    ASSERT_FALSE(emptyDesc->isCompatible(CpuBlockedMemoryDesc{ov::element::undefined, Shape{0}}));
}
