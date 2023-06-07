// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "cache/op_cache.hpp"

namespace {

class OpCacheTest : public ::testing::Test {};

TEST(OpCacheTest, get_op_cache) {
    std::shared_ptr<ov::tools::subgraph_dumper::OpCache> op_cache = nullptr;
    EXPECT_NO_THROW(op_cache = ov::tools::subgraph_dumper::OpCache::get());
    ASSERT_NE(op_cache, nullptr);
}

TEST(OpCacheTest, get_op_cache_twice) {
    std::shared_ptr<ov::tools::subgraph_dumper::OpCache> op_cache_0 = nullptr, op_cache_1 = nullptr;
    op_cache_0 = ov::tools::subgraph_dumper::OpCache::OpCache::get();
    op_cache_1 = ov::tools::subgraph_dumper::OpCache::OpCache::get();
    ASSERT_EQ(op_cache_0, op_cache_1);
}

// TEST_F(OpCacheTest, get_op_cache_twice) {
//     std::shared_ptr<ov::tools::subgraph_dumper::OpCache> op_cache_0 = nullptr, op_cache_1 = nullptr;
//     op_cache_0 = ov::tools::subgraph_dumper::OpCache::OpCache::get();
//     op_cache_1 = ov::tools::subgraph_dumper::OpCache::OpCache::get();
//     ASSERT_EQ(op_cache_0, op_cache_1);
// }

}  // namespace
