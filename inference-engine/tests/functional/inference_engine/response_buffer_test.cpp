// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "description_buffer.hpp"

using namespace std;
using namespace InferenceEngine;

using ResponseBufferTests = ::testing::Test;

TEST_F(ResponseBufferTests, smoke_canCreateResponseMessage) {
    ResponseDesc desc;
    DescriptionBuffer(&desc) << "make error: " << 1;
    ASSERT_STREQ("make error: 1", desc.msg);
}

TEST_F(ResponseBufferTests, smoke_canReportError) {
    ResponseDesc desc;
    DescriptionBuffer d(NETWORK_NOT_LOADED, &desc);
    d << "make error: ";
    ASSERT_EQ(NETWORK_NOT_LOADED, (StatusCode)d);
}

TEST_F(ResponseBufferTests, smoke_savePreviosMessage) {
    ResponseDesc desc;
    desc.msg[0] = 'T';
    desc.msg[1] = 'e';
    desc.msg[2] = 's';
    desc.msg[3] = 't';
    desc.msg[4] = '\0';
    DescriptionBuffer d(&desc);
    ASSERT_EQ(GENERAL_ERROR, (StatusCode)d);
    ASSERT_EQ(std::string("Test"), desc.msg);
}

TEST_F(ResponseBufferTests, smoke_canHandleBigMessage) {
    ResponseDesc desc;
    int size = sizeof(desc.msg) / sizeof(desc.msg[0]);
    DescriptionBuffer buf(&desc);
    std::string bigVal(size, 'A');

    buf << bigVal;
    ASSERT_EQ(desc.msg[0], 'A');
    ASSERT_EQ(desc.msg[size - 2], 'A');
    ASSERT_EQ(desc.msg[size - 1], 0);
}

TEST_F(ResponseBufferTests, smoke_canHandleNotNullTerminatedInput) {
    ResponseDesc desc;
    int size = sizeof(desc.msg) / sizeof(desc.msg[0]);

    desc.msg[size - 1] = 'B';

    DescriptionBuffer buf(&desc);
    std::string bigVal(size, 'A');

    buf << bigVal;
    ASSERT_EQ(desc.msg[0], 'A');
    ASSERT_EQ(desc.msg[size - 2], 'A');
    ASSERT_EQ(desc.msg[size - 1], 0);
}

TEST_F(ResponseBufferTests, smoke_canHandlePredefined) {
    ResponseDesc desc;
    int size = sizeof(desc.msg) / sizeof(desc.msg[0]);

    DescriptionBuffer buf(&desc);
    std::string bigVal(size, 'A');
    buf << bigVal;

    DescriptionBuffer buf2(&desc);
    std::string bigVal2(size, 'B');
    buf2 << bigVal2;

    ASSERT_EQ(desc.msg[0], 'A');
    ASSERT_EQ(desc.msg[size - 2], 'A');
    ASSERT_EQ(desc.msg[size - 1], 0);
}

TEST_F(ResponseBufferTests, smoke_canHandleNotNullTerminatedPredefined) {
    ResponseDesc desc;
    int size = sizeof(desc.msg) / sizeof(desc.msg[0]);

    DescriptionBuffer buf(&desc);
    std::string bigVal(size, 'A');
    buf << bigVal;

    desc.msg[size - 1] = 'B';

    DescriptionBuffer buf2(&desc);
    std::string bigVal2(size, 'B');
    buf2 << bigVal2;

    ASSERT_EQ(desc.msg[0], 'A');
    ASSERT_EQ(desc.msg[size - 2], 'A');
    ASSERT_EQ(desc.msg[size - 1], 0);
}