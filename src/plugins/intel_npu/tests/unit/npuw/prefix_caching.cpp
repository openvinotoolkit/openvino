// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "llm_prefix_caching.hpp"

TEST(PrefixCacheManagerTest, AddAndGetBlock) {
    constexpr size_t cache_capability = 10;
    ov::npuw::PrefixCacheManager cache(cache_capability);

    constexpr size_t block_size = 2;

    auto block1 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block1->set_token_start(0);
    block1->add_block({0x1, 0x2}, {});

    auto block2 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block2->set_token_start(2);
    block2->add_block({0x3, 0x4}, {});

    auto block3 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block3->set_token_start(4);
    block3->add_block({0x5, 0x6}, {});

    cache.put_block(block1, 0x0);
    cache.put_block(block2, block1->get_block_hash());
    cache.put_block(block3, block2->get_block_hash());

    // Retrieve block from cache adn check block hash
    auto retrieved_block = cache.get_block(block1->get_block_hash());
    EXPECT_NE(retrieved_block, nullptr);
    EXPECT_EQ(block1->get_block_hash(), 0x2);

    retrieved_block = cache.get_block(block2->get_block_hash());
    EXPECT_NE(retrieved_block, nullptr);
    EXPECT_EQ(block2->get_block_hash(), 0x4);

    retrieved_block = cache.get_block(block3->get_block_hash());
    EXPECT_NE(retrieved_block, nullptr);
    EXPECT_EQ(block3->get_block_hash(), 0x6);
}

TEST(PrefixCacheManagerTest, LinkBlocks) {
    constexpr size_t cache_capability = 10;
    ov::npuw::PrefixCacheManager cache(cache_capability);

    constexpr size_t block_size = 2;

    auto block1 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block1->add_block({0x1, 0x2}, {});

    auto block2 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block2->add_block({0x3, 0x4}, {});

    cache.put_block(block1, 0);
    cache.put_block(block2, block1->get_block_hash());

    // Check if block1 points to block2
    auto retrieved_block = cache.get_block(block1->get_block_hash());
    EXPECT_NE(retrieved_block, nullptr);
    EXPECT_EQ(retrieved_block->get_child_block_hashes().size(), 1);
    EXPECT_EQ(*retrieved_block->get_child_block_hashes().begin(), block2->get_block_hash());

    // Check if block2 points back to block1
    retrieved_block = cache.get_block(block2->get_block_hash());
    EXPECT_NE(retrieved_block, nullptr);
    EXPECT_EQ(retrieved_block->get_parent_block_hash(), block1->get_block_hash());
}

TEST(PrefixCacheManagerTest, EvictLRUBlock) {
    /*
        Test Purpose: Verify that the PrefixCacheManager's evict_lru_block_unsafe method correctly removes the least
       recently used block.

        Initial Cache State (Capacity is 5):
            1
            |
            2
        /   |   \
        3   4   5
                |
               [6]

        Operation: Attempt to add block 6 to the cache.

        Expected Result: Block 3 is evicted because it is the least recently used and has no child blocks, then block 6
       is successfully added to the cache.

        Final Cache State:
            1
            |
            2
        /   |   \
            4   5
                |
                6
    */
    constexpr size_t cache_capability = 5;
    ov::npuw::PrefixCacheManager cache(cache_capability);

    constexpr size_t block_size = 2;

    auto block1 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block1->add_block({0x1, 0x2}, {});

    auto block2 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block2->add_block({0x3, 0x4}, {});

    auto block3 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block3->add_block({0x5, 0x6}, {});

    auto block4 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block4->add_block({0x7, 0x8}, {});

    auto block5 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block5->add_block({0x9, 0xa}, {});

    cache.put_block(block1, 0);
    cache.put_block(block2, block1->get_block_hash());
    cache.put_block(block3, block2->get_block_hash());
    cache.put_block(block4, block2->get_block_hash());
    cache.put_block(block5, block2->get_block_hash());

    auto block6 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block6->add_block({0xb, 0xc}, {});

    // This should evict block3 as it is the least recently used and has no children
    cache.put_block(block6, block5->get_block_hash());

    EXPECT_NE(cache.get_block(block1->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block2->get_block_hash()), nullptr);

    EXPECT_EQ(cache.get_block(block3->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block4->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block5->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block6->get_block_hash()), nullptr);
}

TEST(PrefixCacheManagerTest, UpdateLRUList) {
    /*
        Test Purpose: Verify that accessing a block updates the LRU list correctly, and ensure the least recently used
       block without children is evicted when the cache is full.

        Initial Cache State (Capacity is 5):
            1
            |
            2
        /   |   \
        3   4   5
                |
               [6]

        Operation:
        1. Access block 3 to update its position in the LRU list.
        2. Attempt to add block 6 to the cache.

        Expected Result:
        - Accessing block 3 updates its position, making block 4 the least recently used.
        - Block 4 is evicted because it is the least recently used and has no child blocks.
        - Block 6 is successfully added to the cache.

        Final Cache State:
            1
            |
            2
        /   |   \
        3       5
                |
                6
    */
    constexpr size_t cache_capability = 5;
    ov::npuw::PrefixCacheManager cache(cache_capability);

    constexpr size_t block_size = 2;

    auto block1 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block1->add_block({0x1, 0x2}, {});

    auto block2 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block2->add_block({0x3, 0x4}, {});

    auto block3 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block3->add_block({0x5, 0x6}, {});

    auto block4 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block4->add_block({0x7, 0x8}, {});

    auto block5 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block5->add_block({0x9, 0xa}, {});

    cache.put_block(block1, 0);
    cache.put_block(block2, block1->get_block_hash());
    cache.put_block(block3, block2->get_block_hash());
    cache.put_block(block4, block2->get_block_hash());
    cache.put_block(block5, block2->get_block_hash());

    // Get block 3 from the cache to update LRU list
    auto retrieved_block = cache.get_block(block3->get_block_hash());
    EXPECT_NE(retrieved_block, nullptr);

    auto block6 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block6->add_block({0xb, 0xc}, {});

    // This should evict block4 as it is the least recently used and has no children
    cache.put_block(block6, block5->get_block_hash());

    EXPECT_NE(cache.get_block(block1->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block2->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block3->get_block_hash()), nullptr);

    EXPECT_EQ(cache.get_block(block4->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block5->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block6->get_block_hash()), nullptr);
}

TEST(PrefixCacheManagerTest, PreventAddingBlock) {
    /*
        Test Purpose: Verify that the cache prevents adding new blocks when the cache is full and no eviction
       candidates are available.

        Initial Cache State (Capacity is 4):
        1 -> 2 -> 3 -> 4

        Operation:
        - Attempt to add block 5 and block 6 to the cache.

        Expected Result:
        - Since the cache is at full capacity and all blocks have child dependencies, there are no eviction candidates.
        - Block 5 and block 6 should not be added to the cache.

        Final Cache State:
        1 -> 2 -> 3 -> 4
        - Block 5 and block 6 are not present in the cache.
    */
    constexpr size_t cache_capability = 4;
    ov::npuw::PrefixCacheManager cache(cache_capability);

    constexpr size_t block_size = 2;

    auto block1 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block1->add_block({0x1, 0x2}, {});

    auto block2 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block2->add_block({0x3, 0x4}, {});

    auto block3 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block3->add_block({0x5, 0x6}, {});

    auto block4 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block4->add_block({0x7, 0x8}, {});

    cache.put_block(block1, 0);
    cache.put_block(block2, block1->get_block_hash());
    cache.put_block(block3, block2->get_block_hash());
    cache.put_block(block4, block3->get_block_hash());

    auto block5 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block5->add_block({0x9, 0xa}, {});

    auto block6 = std::make_shared<ov::npuw::KVBlock>(block_size);
    block6->add_block({0xb, 0xc}, {});

    // Block 5 and block 6 should not be put into the cache successfully
    cache.put_block(block5, block4->get_block_hash());
    cache.put_block(block6, block5->get_block_hash());

    EXPECT_NE(cache.get_block(block1->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block2->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block3->get_block_hash()), nullptr);

    EXPECT_NE(cache.get_block(block4->get_block_hash()), nullptr);

    EXPECT_EQ(cache.get_block(block5->get_block_hash()), nullptr);

    EXPECT_EQ(cache.get_block(block6->get_block_hash()), nullptr);
}
