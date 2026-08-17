// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal_buffer_desc.hpp"

#include <gtest/gtest.h>

using namespace cldnn;

TEST(internal_buffer_desc, carries_runtime_size_alignment_layout_and_lifetime) {
    BufferDescriptor descriptor(256,
                                ov::element::f32,
                                false,
                                true,
                                16,
                                internal_buffer_size_policy::runtime_resolved,
                                internal_buffer_resize_policy::grow_only,
                                internal_buffer_access::read_write);

    EXPECT_NO_THROW(descriptor.validate(256));
    EXPECT_EQ(descriptor.m_layout.bytes_count(), 1024u);
    EXPECT_EQ(descriptor.m_alignment, 16u);
    EXPECT_EQ(descriptor.m_size_policy, internal_buffer_size_policy::runtime_resolved);
    EXPECT_EQ(descriptor.m_lifetime, internal_buffer_lifetime::primitive_instance);
    EXPECT_EQ(descriptor.m_access, internal_buffer_access::read_write);
}

TEST(internal_buffer_desc, distinguishes_grow_only_and_exact_reallocation) {
    BufferDescriptor grow_only(64, ov::element::u8, false, true, 1, internal_buffer_size_policy::runtime_resolved, internal_buffer_resize_policy::grow_only);
    BufferDescriptor exact(64, ov::element::u8, false, true, 1, internal_buffer_size_policy::runtime_resolved, internal_buffer_resize_policy::exact);

    EXPECT_TRUE(grow_only.can_reuse_allocation(128));
    EXPECT_FALSE(exact.can_reuse_allocation(128));
    EXPECT_TRUE(exact.can_reuse_allocation(64));
}

TEST(internal_buffer_desc, rejects_unavailable_alignment) {
    BufferDescriptor descriptor(1, ov::element::u8, false, true, 64);
    EXPECT_THROW(descriptor.validate(16), ov::Exception);
}
