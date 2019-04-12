// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>

#include <gtest/gtest.h>
#include <ie_layouts.h>

using namespace testing;
using namespace InferenceEngine;

class LayoutTest : public ::testing::Test {
public:

};

TEST_F(LayoutTest, offsetNCHW) {
    size_t N = 2, C = 3, H = 4, W = 5;
    SizeVector dims = { W, H, C, N };

    LayoutOffsetCounter loc(InferenceEngine::Layout::NCHW, dims);

    size_t p = loc.Offset({ 0, 0, 0, 0 });
    EXPECT_EQ(p, 0);

    p = loc.Offset({ 1, 1, 1, 1 });
    EXPECT_EQ(p, 86);

    p = loc.Offset({ 4, 3, 2, 1 });
    EXPECT_EQ(p, 119);
}


TEST_F(LayoutTest, offsetNCHW2) {
    size_t N = 1, C = 2, H = 3, W = 2;
    SizeVector dims = { W, H, C, N };

    LayoutOffsetCounter loc(InferenceEngine::Layout::NCHW, dims);

    size_t p = loc.Offset({ 0, 1, 0, 0 });
    EXPECT_EQ(p, 2);

}


TEST_F(LayoutTest, offsetNHWC) {
    size_t N = 1, C = 2, H = 3, W = 2;
    SizeVector dims = { W, H, C, N };

    LayoutOffsetCounter loc(InferenceEngine::Layout::NHWC, dims);

    size_t p = loc.Offset({ 0, 0, 0, 0 });
    EXPECT_EQ(p, 0);

    p = loc.Offset({ 0, 0, 1, 0 });
    EXPECT_EQ(p, 1);

    p = loc.Offset({ 1, 0, 0, 0 });
    EXPECT_EQ(p, 2);

    p = loc.Offset({ 1, 0, 1, 0 });
    EXPECT_EQ(p, 3);

    p = loc.Offset({ 0, 1, 0, 0 });
    EXPECT_EQ(p, 4);

}


TEST_F(LayoutTest, convertNCHWtoNHWC) {
    const size_t N = 1, C = 2, H = 3, W = 2;
    SizeVector dims = { W, H, C, N };

    float sourceBuf[N*C*H*W], destBuf[N*C*H*W], destBuf2[N*C*H*W];
    float p = 0.0;
    for (int nc = 0; nc < N*C; nc++) {
        float v = 0.0;
        for (int hw = 0; hw < H*W; hw++) {
            sourceBuf[nc*H*W + hw] = v + p;
            v++;
        }
        p += 1000;
    }

    ConvertLayout<float>(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NCHW, sourceBuf, destBuf, dims);
    EXPECT_TRUE( 0 == memcmp( sourceBuf, destBuf, N*C*H*W * sizeof(float) ) );

    ConvertLayout(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC, sourceBuf, destBuf, dims);
    ConvertLayout(InferenceEngine::Layout::NHWC, InferenceEngine::Layout::NCHW, destBuf, destBuf2, dims);
    EXPECT_TRUE( 0 == memcmp( sourceBuf, destBuf2, N*C*H*W * sizeof(float) ) );
}

