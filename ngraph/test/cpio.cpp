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

#include <memory>

#include <gtest/gtest.h>

#include "ngraph/cpio.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"

using namespace ngraph;
using namespace std;

TEST(cpio, read)
{
    const string test_file = file_util::path_join(TEST_FILES, "test.cpio");

    cpio::Reader reader(test_file);
    auto file_info = reader.get_file_info();
    ASSERT_EQ(3, file_info.size());
    EXPECT_STREQ(file_info[0].get_name().c_str(), "test1.txt");
    EXPECT_STREQ(file_info[1].get_name().c_str(), "test2.txt");
    EXPECT_STREQ(file_info[2].get_name().c_str(), "test3.txt");

    EXPECT_EQ(file_info[0].get_size(), 5);
    EXPECT_EQ(file_info[1].get_size(), 14);
    EXPECT_EQ(file_info[2].get_size(), 44);

    {
        int index = 0;
        runtime::AlignedBuffer data(file_info[index].get_size());
        reader.read(file_info[index].get_name(), data.get_ptr(), file_info[index].get_size());
        string content = string(data.get_ptr<char>(), file_info[index].get_size());
        EXPECT_STREQ(content.c_str(), "12345");
    }

    {
        int index = 1;
        runtime::AlignedBuffer data(file_info[index].get_size());
        reader.read(file_info[index].get_name(), data.get_ptr(), file_info[index].get_size());
        string content = string(data.get_ptr<char>(), file_info[index].get_size());
        EXPECT_STREQ(content.c_str(), "this is a test");
    }

    {
        int index = 2;
        runtime::AlignedBuffer data(file_info[index].get_size());
        reader.read(file_info[index].get_name(), data.get_ptr(), file_info[index].get_size());
        string content = string(data.get_ptr<char>(), file_info[index].get_size());
        EXPECT_STREQ(content.c_str(), "the quick brown fox jumped over the lazy dog");
    }
}

TEST(cpio, write)
{
    const string test_file = "test1.cpio";
    string s1 = "this is a test";
    string s2 = "the quick brown fox jumps over the lazy dog";
    {
        cpio::Writer writer(test_file);
        {
            writer.write("file1.txt", s1.data(), static_cast<uint32_t>(s1.size()));
        }
        {
            writer.write("file.txt", s2.data(), static_cast<uint32_t>(s2.size()));
        }
    }
    {
        cpio::Reader reader(test_file);
        auto file_info = reader.get_file_info();
        ASSERT_EQ(2, file_info.size());

        EXPECT_STREQ(file_info[0].get_name().c_str(), "file1.txt");
        EXPECT_STREQ(file_info[1].get_name().c_str(), "file.txt");

        EXPECT_EQ(file_info[0].get_size(), 14);
        EXPECT_EQ(file_info[1].get_size(), 43);

        {
            int index = 0;
            runtime::AlignedBuffer data(file_info[index].get_size());
            reader.read(file_info[index].get_name(), data.get_ptr(), file_info[index].get_size());
            string content = string(data.get_ptr<char>(), file_info[index].get_size());
            EXPECT_STREQ(content.c_str(), s1.c_str());
        }

        {
            int index = 1;
            runtime::AlignedBuffer data(file_info[index].get_size());
            reader.read(file_info[index].get_name(), data.get_ptr(), file_info[index].get_size());
            string content = string(data.get_ptr<char>(), file_info[index].get_size());
            EXPECT_STREQ(content.c_str(), s2.c_str());
        }
    }
}
