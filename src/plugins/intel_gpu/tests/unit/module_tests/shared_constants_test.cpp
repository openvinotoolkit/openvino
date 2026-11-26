// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "intel_gpu/plugin/remote_context.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

#include <memory>
#include <vector>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

namespace {
struct shared_constants_test : public ::testing::Test {
    void SetUp() override {
        if (!get_test_engine().supports_allocation(allocation_type::usm_device))
            GTEST_SKIP() << "usm_device is not supported";
        context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{get_test_engine().get_device()});
    }

    // Allocates the constant memory and counts how often the context asked for it
    memory::ptr get(const std::shared_ptr<ov::AlignedBuffer>& source, ov::weight_sharing::DataID constant_id, const layout& l) {
        return context->get_or_create_shared_constant(source, constant_id, l, [&]() {
            ++created;
            return context->get_engine().allocate_memory(l, allocation_type::usm_device, false);
        });
    }

    std::shared_ptr<RemoteContextImpl> context;
    layout constant_layout{ov::PartialShape{16, 32}, data_types::f16, format::bfyx};
    size_t created = 0;
};
}  // namespace

TEST_F(shared_constants_test, same_constant_of_same_source_gets_one_memory) {
    auto source = std::make_shared<ov::AlignedBuffer>(4096, 64);

    auto first = get(source, 1, constant_layout);
    auto second = get(source, 1, constant_layout);

    ASSERT_EQ(first, second);
    ASSERT_EQ(created, 1);
}

TEST_F(shared_constants_test, other_constant_id_or_layout_gets_own_memory) {
    auto source = std::make_shared<ov::AlignedBuffer>(4096, 64);
    layout other_layout{ov::PartialShape{32, 16}, data_types::f16, format::bfyx};

    auto first = get(source, 1, constant_layout);
    auto other_id = get(source, 2, constant_layout);
    auto other_layout_mem = get(source, 1, other_layout);

    ASSERT_NE(first, other_id);
    ASSERT_NE(first, other_layout_mem);
    ASSERT_EQ(created, 3);
}

// Source ids are not unique across producers, and a freed source address can be reused, so only the owner identifies a source
TEST_F(shared_constants_test, other_owner_at_same_address_gets_own_memory) {
    auto source = std::make_shared<ov::AlignedBuffer>(4096, 64);
    auto other_owner = std::make_shared<int>(0);
    std::shared_ptr<ov::AlignedBuffer> same_address(other_owner, source.get());

    auto first = get(source, 1, constant_layout);
    auto second = get(same_address, 1, constant_layout);

    ASSERT_NE(first, second);
    ASSERT_EQ(created, 2);
}

// The cache must not keep device memory alive once no compiled model uses it
TEST_F(shared_constants_test, memory_is_released_with_its_last_user) {
    auto source = std::make_shared<ov::AlignedBuffer>(4096, 64);

    auto first = get(source, 1, constant_layout);
    std::weak_ptr<memory> released = first;
    first.reset();
    ASSERT_TRUE(released.expired());

    auto second = get(source, 1, constant_layout);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(created, 2);
}
