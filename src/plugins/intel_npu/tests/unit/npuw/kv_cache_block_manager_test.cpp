// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_block_manager.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace ov::npuw;

namespace {

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

}  // namespace
