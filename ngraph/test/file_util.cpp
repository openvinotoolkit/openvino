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

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"

using namespace std;
using namespace ngraph;

TEST(file_util, path_join)
{
    {
        string s1 = "";
        string s2 = "";

        EXPECT_STREQ("", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "";
        string s2 = "/test1/test2";

        EXPECT_STREQ("/test1/test2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "";
        string s2 = "/test1/test2/";

        EXPECT_STREQ("/test1/test2/", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "";
        string s2 = "test1/test2";

        EXPECT_STREQ("test1/test2", file_util::path_join(s1, s2).c_str());
    }

    {
        string s1 = "/x1/x2";
        string s2 = "";

        EXPECT_STREQ("/x1/x2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "/";

        EXPECT_STREQ("/", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/x1/x2";
        string s2 = "/test1/test2";

        EXPECT_STREQ("/test1/test2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "test1/test2";

        EXPECT_STREQ("/x1/x2/test1/test2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/x1/x2";
        string s2 = "test1/test2";

        EXPECT_STREQ("/x1/x2/test1/test2", file_util::path_join(s1, s2).c_str());
    }
    {
        string s1 = "/";
        string s2 = "test1/test2";

        EXPECT_STREQ("/test1/test2", file_util::path_join(s1, s2).c_str());
    }
}

TEST(file_util, get_temp_directory_path)
{
    string tmp = file_util::get_temp_directory_path();
    EXPECT_NE(0, tmp.size());
}
