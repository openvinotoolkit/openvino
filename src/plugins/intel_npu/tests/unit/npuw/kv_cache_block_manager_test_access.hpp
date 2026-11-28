// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kv_cache_block_manager.hpp"

namespace ov {
namespace test {
namespace npuw {

// Reaches the strategy-only entry points and the state of the block manager.
struct KVCacheBlockManagerTestAccess {
    static uint32_t inject(ov::npuw::KVCacheBlockManager& manager, ov::SoPtr<ov::ITensor> tensor, uint32_t tokens) {
        return manager.allocate_block_with_tensor(std::move(tensor), tokens);
    }
    static void share(ov::npuw::KVCacheBlockManager& manager, uint32_t block_id) {
        manager.mark_block_shared(block_id);
    }
    static bool make_private(ov::npuw::KVCacheBlockManager& manager, uint32_t block_id) {
        return manager.make_block_private(block_id);
    }
    static bool is_shared(const ov::npuw::KVCacheBlockManager& manager, uint32_t block_id) {
        return manager.blocks_.at(block_id).is_shared;
    }
};

}  // namespace npuw
}  // namespace test
}  // namespace ov
