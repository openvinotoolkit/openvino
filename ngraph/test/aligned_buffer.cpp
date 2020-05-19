//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"

#include "ngraph/runtime/aligned_buffer.hpp"

using namespace std;
using namespace ngraph;

TEST(aligned_buffer, alignment)
{
    runtime::AlignedBuffer buffer(100, 64);
    size_t addr = reinterpret_cast<size_t>(buffer.get_ptr()) % 64;
    EXPECT_EQ(addr, 0);
}

TEST(aligned_buffer, move)
{
    {
        runtime::AlignedBuffer buffer1(100, 64);
        runtime::AlignedBuffer buffer2(move(buffer1));
        EXPECT_EQ(buffer1.size(), 0);
        EXPECT_EQ(buffer1.get_ptr(), nullptr);
        EXPECT_EQ(buffer2.size(), 100);
        EXPECT_NE(buffer2.get_ptr(), nullptr);
    }
    {
        runtime::AlignedBuffer buffer1(100, 64);
        runtime::AlignedBuffer buffer2;
        buffer2 = move(buffer1);
        EXPECT_EQ(buffer1.size(), 0);
        EXPECT_EQ(buffer1.get_ptr(), nullptr);
        EXPECT_EQ(buffer2.size(), 100);
        EXPECT_NE(buffer2.get_ptr(), nullptr);
    }
}
