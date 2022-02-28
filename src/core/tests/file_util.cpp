// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/file_util.hpp"

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(file_util, path_join) {
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

TEST(file_util, santize_path) {
    {
        string path = "../../tensor.data";
        EXPECT_STREQ("tensor.data", file_util::sanitize_path(path).c_str());
    }
    {
        string path = "/../tensor.data";
        EXPECT_STREQ("tensor.data", file_util::sanitize_path(path).c_str());
    }
    {
        string path = "..";
        EXPECT_STREQ("", file_util::sanitize_path(path).c_str());
    }
    {
        string path = "workspace/data/tensor.data";
        EXPECT_STREQ("workspace/data/tensor.data", file_util::sanitize_path(path).c_str());
    }
    {
        string path = "..\\..\\tensor.data";
        EXPECT_STREQ("tensor.data", file_util::sanitize_path(path).c_str());
    }
    {
        string path = "C:\\workspace\\tensor.data";
        EXPECT_STREQ("workspace\\tensor.data", file_util::sanitize_path(path).c_str());
    }
}
