// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "blocking_queue.hpp"

using namespace XLinkPro;

class BlockingQueueTestBase : public ::testing::Test {
};

TEST_F(BlockingQueueTestBase, CreateBlockingQueue) {
    BlockingQueue<int> q = {};
}