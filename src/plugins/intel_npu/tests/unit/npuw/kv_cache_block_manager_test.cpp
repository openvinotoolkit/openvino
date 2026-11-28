// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_block_manager.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>

#include "kv_cache_block_manager_test_access.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "util.hpp"

using namespace ov::npuw;

namespace {

using ov::test::npuw::KVCacheBlockManagerTestAccess;

// Test fixture for KVCacheBlockManager tests
class KVCacheBlockManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test parameters
        block_size = 512;
        max_blocks = 8;
        base_shape = ov::Shape{1, 32, block_size, 128};  // [batch, num_heads, seq_len=block_size, head_dim]
        elem_type = ov::element::f16;
        device = "CPU";  // Use CPU for unit tests

        // Pass nullptr for plugin since CPU device doesn't need it in allocMem
        manager = std::make_unique<KVCacheBlockManager>(block_size, max_blocks, base_shape, elem_type, device, nullptr);
    }

    void TearDown() override {
        manager.reset();
    }

    uint32_t block_size;
    uint32_t max_blocks;
    ov::Shape base_shape;
    ov::element::Type elem_type;
    std::string device;
    std::unique_ptr<KVCacheBlockManager> manager;
};

// Test 1: Basic Allocation
TEST_F(KVCacheBlockManagerTest, BasicAllocation) {
    // Allocate a block
    auto block_id = manager->allocate_block();
    ASSERT_TRUE(block_id.has_value()) << "Block allocation should succeed";
    EXPECT_EQ(block_id.value(), 0u) << "First block should have ID 0";

    // Get block tensor
    auto tensor = manager->get_block_tensor(block_id.value());
    ASSERT_NE(tensor, nullptr) << "Block tensor should not be null";

    // Check tensor shape
    auto shape = tensor->get_shape();
    EXPECT_EQ(shape[0], base_shape[0]) << "Batch dimension should match";
    EXPECT_EQ(shape[1], base_shape[1]) << "Num_heads dimension should match";
    EXPECT_EQ(shape[2], block_size) << "Seq_len should be block_size";
    EXPECT_EQ(shape[3], base_shape[3]) << "Head_dim should match";

    // Check token count
    EXPECT_EQ(manager->get_block_tokens(block_id.value()), 0u) << "Newly allocated block should have 0 tokens";
}

// Test 2: Multiple Allocations
TEST_F(KVCacheBlockManagerTest, MultipleAllocations) {
    std::vector<uint32_t> allocated_blocks;

    // Allocate half of available blocks
    for (uint32_t i = 0; i < max_blocks / 2; ++i) {
        auto block_id = manager->allocate_block();
        ASSERT_TRUE(block_id.has_value()) << "Allocation " << i << " should succeed";
        allocated_blocks.push_back(block_id.value());
    }

    // Verify unique block IDs
    std::set<uint32_t> unique_ids(allocated_blocks.begin(), allocated_blocks.end());
    EXPECT_EQ(unique_ids.size(), allocated_blocks.size()) << "All allocated blocks should have unique IDs";

    // Verify counts via get_allocated_blocks()
    auto all_allocated = manager->get_allocated_blocks();
    EXPECT_EQ(all_allocated.size(), max_blocks / 2);
}

// Test 3: Block Exhaustion
TEST_F(KVCacheBlockManagerTest, BlockExhaustion) {
    // Allocate all blocks
    for (uint32_t i = 0; i < max_blocks; ++i) {
        auto block_id = manager->allocate_block();
        ASSERT_TRUE(block_id.has_value()) << "Allocation " << i << " should succeed";
    }

    // All blocks should be allocated
    EXPECT_EQ(manager->get_allocated_blocks().size(), static_cast<size_t>(max_blocks));

    // Try to allocate one more - should fail
    auto extra_block = manager->allocate_block();
    EXPECT_FALSE(extra_block.has_value()) << "Allocation should fail when all blocks are in use";

    // Clear all, then allocation should succeed again
    manager->clear_all();
    EXPECT_TRUE(manager->get_allocated_blocks().empty());

    auto new_block = manager->allocate_block();
    ASSERT_TRUE(new_block.has_value()) << "Allocation should succeed after clearing all blocks";
}

// Test 4: Token Tracking
TEST_F(KVCacheBlockManagerTest, TokenTracking) {
    auto block_id = manager->allocate_block();
    ASSERT_TRUE(block_id.has_value());

    // Initially 0 tokens
    EXPECT_EQ(manager->get_block_tokens(block_id.value()), 0u);

    // Update to 256 tokens
    manager->update_block_tokens(block_id.value(), 256);
    EXPECT_EQ(manager->get_block_tokens(block_id.value()), 256u);

    // Update to full capacity
    manager->update_block_tokens(block_id.value(), block_size);
    EXPECT_EQ(manager->get_block_tokens(block_id.value()), block_size);

    // Allocate a second block and verify its tokens are tracked independently
    auto block_id2 = manager->allocate_block();
    ASSERT_TRUE(block_id2.has_value());
    manager->update_block_tokens(block_id2.value(), 100);
    EXPECT_EQ(manager->get_block_tokens(block_id2.value()), 100u);
    EXPECT_EQ(manager->get_block_tokens(block_id.value()), block_size) << "First block tokens should be unchanged";
}

// Test 5: Token Count Validation
TEST_F(KVCacheBlockManagerTest, TokenCountValidation) {
    auto block_id = manager->allocate_block();
    ASSERT_TRUE(block_id.has_value());

    // Try to set more tokens than capacity - should throw
    EXPECT_THROW(manager->update_block_tokens(block_id.value(), block_size + 1), ov::Exception)
        << "Setting tokens beyond capacity should throw exception";
}

// Test 6: Tensor Valid After Clear
// clear_all() drops all SoPtr tensors (releases device memory back to the allocator).
// Re-allocation must produce a fresh valid tensor; the address is not guaranteed to be reused.
TEST_F(KVCacheBlockManagerTest, TensorValidAfterClear) {
    auto block_id = manager->allocate_block();
    ASSERT_TRUE(block_id.has_value());
    ASSERT_NE(manager->get_block_tensor(block_id.value()), nullptr);

    // clear_all() releases tensor memory; block state is reset
    manager->clear_all();
    EXPECT_TRUE(manager->get_allocated_blocks().empty());

    // Re-allocate: a new tensor must be allocated on demand
    auto block_id2 = manager->allocate_block();
    ASSERT_TRUE(block_id2.has_value());
    auto tensor_after = manager->get_block_tensor(block_id2.value());
    ASSERT_NE(tensor_after, nullptr) << "Fresh tensor must be valid after clear + re-allocate";
    EXPECT_EQ(manager->get_block_tokens(block_id2.value()), 0u) << "Token count must be reset to 0";
}

// Test 7: Allocation and Token State Consistency
TEST_F(KVCacheBlockManagerTest, AllocationAndTokenConsistency) {
    // Initially no blocks allocated
    EXPECT_TRUE(manager->get_allocated_blocks().empty());

    // Allocate 2 blocks with different token counts
    auto block1 = manager->allocate_block();
    auto block2 = manager->allocate_block();
    ASSERT_TRUE(block1.has_value());
    ASSERT_TRUE(block2.has_value());

    manager->update_block_tokens(block1.value(), block_size);  // full block
    manager->update_block_tokens(block2.value(), 256);         // half-full block

    EXPECT_EQ(manager->get_allocated_blocks().size(), 2u);
    EXPECT_EQ(manager->get_block_tokens(block1.value()), block_size);
    EXPECT_EQ(manager->get_block_tokens(block2.value()), 256u);

    // After clear, tokens and allocations reset
    manager->clear_all();
    EXPECT_TRUE(manager->get_allocated_blocks().empty());
    // Re-allocate and verify tokens are reset to 0
    auto block3 = manager->allocate_block();
    ASSERT_TRUE(block3.has_value());
    EXPECT_EQ(manager->get_block_tokens(block3.value()), 0u);
}

// Test 8: Clear All
TEST_F(KVCacheBlockManagerTest, ClearAll) {
    // Allocate several blocks with token counts
    for (uint32_t i = 0; i < 5; ++i) {
        auto block_id = manager->allocate_block();
        ASSERT_TRUE(block_id.has_value());
        manager->update_block_tokens(block_id.value(), 100 * (i + 1));
    }

    // Verify blocks allocated
    EXPECT_EQ(manager->get_allocated_blocks().size(), 5u);

    // Clear all
    manager->clear_all();

    // Verify all blocks freed
    EXPECT_TRUE(manager->get_allocated_blocks().empty());

    // Should be able to allocate again, and tokens reset to 0
    auto new_block = manager->allocate_block();
    ASSERT_TRUE(new_block.has_value());
    EXPECT_EQ(manager->get_block_tokens(new_block.value()), 0u);
}

// Test 9: Get Allocated Blocks
TEST_F(KVCacheBlockManagerTest, GetAllocatedBlocks) {
    // Initially empty
    auto allocated = manager->get_allocated_blocks();
    EXPECT_TRUE(allocated.empty());

    // Allocate some blocks
    std::vector<uint32_t> expected_blocks;
    expected_blocks.push_back(manager->allocate_block().value());
    expected_blocks.push_back(manager->allocate_block().value());
    expected_blocks.push_back(manager->allocate_block().value());

    // Get allocated blocks
    allocated = manager->get_allocated_blocks();
    EXPECT_EQ(allocated.size(), expected_blocks.size());

    // Verify all expected blocks are in the list
    for (auto block_id : expected_blocks) {
        EXPECT_TRUE(std::find(allocated.begin(), allocated.end(), block_id) != allocated.end())
            << "Block " << block_id << " should be in allocated list";
    }
}

// Test 10: Invalid Block ID
TEST_F(KVCacheBlockManagerTest, InvalidBlockID) {
    // Try to access block with ID >= max_blocks
    EXPECT_THROW(manager->get_block_tensor(max_blocks), ov::Exception) << "Accessing invalid block ID should throw";

    EXPECT_THROW(manager->get_block_tokens(max_blocks + 10), ov::Exception)
        << "Accessing invalid block ID should throw";

    EXPECT_THROW(manager->update_block_tokens(max_blocks, 100), ov::Exception)
        << "Updating invalid block ID should throw";
}

// Truncation for continuous prefill keeps the leading blocks and frees the suffix.
TEST_F(KVCacheBlockManagerTest, TruncateAllocatedKeepsPrefix) {
    std::vector<uint32_t> ids;
    for (int i = 0; i < 5; ++i) {
        auto id = manager->allocate_block();
        ASSERT_TRUE(id.has_value());
        manager->update_block_tokens(id.value(), block_size);
        ids.push_back(id.value());
    }

    manager->truncate_allocated(2);

    auto allocated = manager->get_allocated_blocks();
    ASSERT_EQ(allocated.size(), 2u);
    EXPECT_EQ(allocated[0], ids[0]);
    EXPECT_EQ(allocated[1], ids[1]);
    // Retained blocks keep their token counts and tensors.
    EXPECT_EQ(manager->get_block_tokens(ids[0]), block_size);
    EXPECT_EQ(manager->get_block_tokens(ids[1]), block_size);
    EXPECT_NE(manager->get_block_tensor(ids[0]), nullptr);
    EXPECT_EQ(manager->num_free_blocks(), max_blocks - 2);
}

TEST_F(KVCacheBlockManagerTest, TruncateAllocatedPreservesAppendOrder) {
    std::vector<uint32_t> ids;
    for (int i = 0; i < 4; ++i) {
        auto id = manager->allocate_block();
        ASSERT_TRUE(id.has_value());
        ids.push_back(id.value());
    }

    manager->truncate_allocated(1);

    // Freed suffix IDs come back in ascending order so logical append order holds.
    auto first = manager->allocate_block();
    auto second = manager->allocate_block();
    ASSERT_TRUE(first.has_value() && second.has_value());
    EXPECT_EQ(first.value(), ids[1]);
    EXPECT_EQ(second.value(), ids[2]);
    // Reacquired blocks start empty.
    EXPECT_EQ(manager->get_block_tokens(first.value()), 0u);
}

TEST_F(KVCacheBlockManagerTest, TruncateAllocatedKeepsSuffixTensorsWarm) {
    std::vector<uint32_t> ids;
    std::vector<void*> data_before;
    for (int i = 0; i < 3; ++i) {
        auto id = manager->allocate_block();
        ASSERT_TRUE(id.has_value());
        ids.push_back(id.value());
        data_before.push_back(manager->get_block_tensor(id.value())->data());
    }

    manager->truncate_allocated(1);

    // Retained prefix keeps its tensor untouched.
    EXPECT_EQ(manager->get_block_tensor(ids[0])->data(), data_before[0]);

    // Suffix tensors are kept warm, so reacquiring those IDs must not re-allocate.
    // Without this the first prefill chunk of a continuation pays allocation latency.
    for (size_t i = 1; i < ids.size(); ++i) {
        auto reacquired = manager->allocate_block();
        ASSERT_TRUE(reacquired.has_value());
        EXPECT_EQ(reacquired.value(), ids[i]);
        EXPECT_EQ(manager->get_block_tensor(reacquired.value())->data(), data_before[i]);
    }
}

TEST_F(KVCacheBlockManagerTest, TruncateAllocatedRejectsOverKeep) {
    ASSERT_TRUE(manager->allocate_block().has_value());
    EXPECT_THROW(manager->truncate_allocated(2), ov::Exception);
    // Truncating to the full allocated count is a no-op.
    manager->truncate_allocated(1);
    EXPECT_EQ(manager->get_allocated_blocks().size(), 1u);
}

// Test 11: Get Tensor After Clear (memory kept, state reset)
TEST_F(KVCacheBlockManagerTest, GetTensorAfterClear) {
    auto block_id = manager->allocate_block();
    ASSERT_TRUE(block_id.has_value());

    // Tensor is allocated on first allocate_block()
    auto tensor = manager->get_block_tensor(block_id.value());
    EXPECT_NE(tensor, nullptr) << "Tensor should be valid after allocation";

    // clear_all() resets state but keeps memory for reuse
    manager->clear_all();
    auto block_id2 = manager->allocate_block();
    ASSERT_TRUE(block_id2.has_value());
    auto tensor2 = manager->get_block_tensor(block_id2.value());
    EXPECT_NE(tensor2, nullptr) << "Tensor should still be valid after clear + re-allocate";
}

// Test 12: Invalid Constructor Parameters
TEST_F(KVCacheBlockManagerTest, InvalidConstructorParams) {
    const ov::Shape valid_shape{1, 32, 512, 128};

    // block_size = 0 must throw
    EXPECT_THROW(KVCacheBlockManager(0u, 8u, valid_shape, ov::element::f16, "CPU", nullptr), ov::Exception)
        << "block_size=0 must throw";

    // max_blocks = 0 must throw
    EXPECT_THROW(KVCacheBlockManager(512u, 0u, valid_shape, ov::element::f16, "CPU", nullptr), ov::Exception)
        << "max_blocks=0 must throw";

    // Shape with no dimension equal to block_size must throw
    const ov::Shape bad_shape{1, 32, 256, 128};  // neither dim[2]=256 nor dim[3]=128 equals 512
    EXPECT_THROW(KVCacheBlockManager(512u, 8u, bad_shape, ov::element::f16, "CPU", nullptr), ov::Exception)
        << "Shape not containing block_size must throw";
}

// Test 13: update_block_tokens on a Freed Block
TEST_F(KVCacheBlockManagerTest, UpdateTokensOnFreeBlock) {
    auto block_id = manager->allocate_block();
    ASSERT_TRUE(block_id.has_value());
    manager->update_block_tokens(block_id.value(), 100);

    // clear_all() marks block as free
    manager->clear_all();

    // Updating a free (is_allocated=false) block must throw
    EXPECT_THROW(manager->update_block_tokens(block_id.value(), 100), ov::Exception)
        << "update_block_tokens on a freed block must throw";
}

// Test 14: num_free_blocks Accuracy
TEST_F(KVCacheBlockManagerTest, NumFreeBlocks) {
    // Initially all blocks are free
    EXPECT_EQ(manager->num_free_blocks(), max_blocks);

    // Each allocation reduces the free count by 1
    manager->allocate_block();
    manager->allocate_block();
    manager->allocate_block();
    EXPECT_EQ(manager->num_free_blocks(), max_blocks - 3u);

    // clear_all() restores all blocks to free
    manager->clear_all();
    EXPECT_EQ(manager->num_free_blocks(), max_blocks);

    // Exhaust the pool: free count must reach 0
    for (uint32_t i = 0; i < max_blocks; ++i) {
        manager->allocate_block();
    }
    EXPECT_EQ(manager->num_free_blocks(), 0u);
}

// Injected blocks take the smallest free IDs, so block ID order stays token order.
TEST_F(KVCacheBlockManagerTest, InjectedBlocksKeepSequenceOrder) {
    auto first = ov::npuw::util::allocMem(elem_type, base_shape, device, nullptr);
    auto second = ov::npuw::util::allocMem(elem_type, base_shape, device, nullptr);

    const auto id0 = KVCacheBlockManagerTestAccess::inject(*manager, first, block_size);
    const auto id1 = KVCacheBlockManagerTestAccess::inject(*manager, second, block_size);
    const auto id2 = manager->allocate_block().value();

    const auto allocated = manager->get_allocated_blocks();
    ASSERT_EQ(allocated.size(), 3u);
    EXPECT_EQ(allocated[0], id0);
    EXPECT_EQ(allocated[1], id1);
    EXPECT_EQ(allocated[2], id2);
    EXPECT_EQ(manager->get_block_tensor(id0)->data(), first->data());
    EXPECT_EQ(manager->get_block_tensor(id1)->data(), second->data());
    EXPECT_EQ(manager->get_block_tokens(id0), block_size);
}

// A tensor the prefix cache references must not stay in the pool as warm scratch memory.
TEST_F(KVCacheBlockManagerTest, ReleaseDropsSharedTensorsEvenWhenWarm) {
    const auto shared_id = manager->allocate_block().value();
    const auto private_id = manager->allocate_block().value();
    auto shared_tensor = manager->get_block_tensor(shared_id);
    auto private_data = manager->get_block_tensor(private_id)->data();
    KVCacheBlockManagerTestAccess::share(*manager, shared_id);

    manager->release(2);

    // The pool let go of the shared tensor, so only this test still holds it.
    EXPECT_EQ(shared_tensor._ptr.use_count(), 1);
    // The private block was kept warm and is handed out again with the same memory.
    EXPECT_EQ(manager->allocate_block().value(), shared_id);
    EXPECT_NE(manager->get_block_tensor(shared_id)->data(), shared_tensor->data());
    EXPECT_EQ(manager->allocate_block().value(), private_id);
    EXPECT_EQ(manager->get_block_tensor(private_id)->data(), private_data);
}

TEST_F(KVCacheBlockManagerTest, TruncateDropsSharedSuffixTensors) {
    manager->allocate_block();
    const auto shared_id = manager->allocate_block().value();
    auto shared_tensor = manager->get_block_tensor(shared_id);
    KVCacheBlockManagerTestAccess::share(*manager, shared_id);

    manager->truncate_allocated(1);

    EXPECT_EQ(shared_tensor._ptr.use_count(), 1);
}

TEST_F(KVCacheBlockManagerTest, SharedFlagFollowsTheBlockLifetime) {
    const auto id = manager->allocate_block().value();
    EXPECT_FALSE(KVCacheBlockManagerTestAccess::is_shared(*manager, id));

    KVCacheBlockManagerTestAccess::share(*manager, id);
    EXPECT_TRUE(KVCacheBlockManagerTestAccess::is_shared(*manager, id));

    // A block handed out again after a release is private until it is shared once more.
    manager->release(max_blocks);
    ASSERT_EQ(manager->allocate_block().value(), id);
    EXPECT_FALSE(KVCacheBlockManagerTestAccess::is_shared(*manager, id));

    // An injected tensor is shared from the start.
    auto cached = ov::npuw::util::allocMem(elem_type, base_shape, device, nullptr);
    const auto injected_id = KVCacheBlockManagerTestAccess::inject(*manager, cached, block_size);
    EXPECT_TRUE(KVCacheBlockManagerTestAccess::is_shared(*manager, injected_id));
}

TEST_F(KVCacheBlockManagerTest, MakePrivateGivesPrivateCopy) {
    const auto id = manager->allocate_block().value();
    auto original = manager->get_block_tensor(id);
    std::fill_n(original->data<ov::float16>(), 8, ov::float16(3.0f));
    manager->update_block_tokens(id, block_size);

    // A block that was never shared is left alone.
    EXPECT_FALSE(KVCacheBlockManagerTestAccess::make_private(*manager, id));
    EXPECT_EQ(manager->get_block_tensor(id)->data(), original->data());

    KVCacheBlockManagerTestAccess::share(*manager, id);
    EXPECT_TRUE(KVCacheBlockManagerTestAccess::make_private(*manager, id));

    auto copy = manager->get_block_tensor(id);
    EXPECT_NE(copy->data(), original->data());
    EXPECT_EQ(copy->data<ov::float16>()[0], ov::float16(3.0f));
    EXPECT_EQ(manager->get_block_tokens(id), block_size);
}

}  // namespace
