// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_util.hpp"

#include <gtest/gtest.h>

#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace ov;

TEST(file_util, path_join) {
    {
        string s1 = "";
        string s2 = "";

        EXPECT_EQ(ov::util::Path{""}, ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "";
        string s2 = "/test1/test2";

        EXPECT_EQ(ov::util::Path{"/test1/test2"}, ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "";
        string s2 = "/test1/test2/";

        EXPECT_EQ(ov::util::Path{"/test1/test2/"}, ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "";
        string s2 = "test1/test2";

        EXPECT_EQ(ov::util::Path{"test1/test2"}, ov::util::path_join({s1, s2}));
    }

    {
        string s1 = "/x1/x2";
        string s2 = "";

        EXPECT_EQ(ov::util::Path{"/x1/x2"}, ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "/";

        EXPECT_EQ(ov::util::Path{"/"}, ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "/x1/x2";
        string s2 = "/test1/test2";

        EXPECT_EQ(ov::util::Path{"/test1/test2"}, ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "test1/test2";

        EXPECT_EQ(ov::util::Path{"/x1/x2/test1/test2"}, ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "/x1/x2";
        string s2 = "test1/test2";

#ifndef _WIN32
        EXPECT_EQ(ov::util::Path{"/x1/x2/test1/test2"}, ov::util::path_join({s1, s2}));
#else
        EXPECT_EQ(ov::util::Path{"/x1/x2\\test1/test2"}, ov::util::path_join({s1, s2}));
#endif
    }
    {
        string s1 = "/";
        string s2 = "test1/test2";

        EXPECT_EQ(ov::util::Path{"/test1/test2"}, ov::util::path_join({s1, s2}));
    }
}

TEST(file_util, sanitize_path) {
    {
        string path = "../../tensor.data";
        EXPECT_STREQ("tensor.data", ov::util::sanitize_path(path).c_str());
    }
    {
        string path = "/../tensor.data";
        EXPECT_STREQ("tensor.data", ov::util::sanitize_path(path).c_str());
    }
    {
        string path = "..";
        EXPECT_STREQ("", ov::util::sanitize_path(path).c_str());
    }
    {
        string path = "workspace/data/tensor.data";
        EXPECT_STREQ("workspace/data/tensor.data", ov::util::sanitize_path(path).c_str());
    }
    {
        string path = "..\\..\\tensor.data";
        EXPECT_STREQ("tensor.data", ov::util::sanitize_path(path).c_str());
    }
    {
        string path = "C:\\workspace\\tensor.data";
        EXPECT_STREQ("workspace\\tensor.data", ov::util::sanitize_path(path).c_str());
    }
}

using namespace testing;

class TrimFileTest : public Test {
protected:
    void SetUp() override {
        project_dir_name = std::string(OV_NATIVE_PARENT_PROJECT_ROOT_DIR);
    }

    std::string project_dir_name;
};

TEST_F(TrimFileTest, relative_path_to_source) {
    const auto exp_path = ov::util::path_join({"src", "test_src.cpp"});

    const auto file_path = ov::util::path_join({"..", "..", "..", project_dir_name, "src", "test_src.cpp"});

    auto str_ptr = ov::util::trim_file_name(file_path);
    EXPECT_EQ(exp_path, str_ptr);
}

TEST_F(TrimFileTest, relative_path_to_source_but_no_project_dir) {
    const auto file_path = ov::util::path_join({"..", "..", "..", "src", "test_src.cpp"});

    auto str_ptr = ov::util::trim_file_name(file_path);
    EXPECT_EQ(file_path, str_ptr);
}

TEST_F(TrimFileTest, absolute_path_to_source) {
    const auto exp_path = ov::util::path_join({"src", "test_src.cpp"});

    const auto file_path = ov::util::path_join({"home", "user", project_dir_name, "src", "test_src.cpp"});

    auto str_ptr = ov::util::trim_file_name(file_path);
    EXPECT_EQ(exp_path, str_ptr);
}

TEST_F(TrimFileTest, absolute_path_to_source_but_no_project_dir) {
    const auto file_path = ov::util::path_join({"home", "user", "src", "test_src.cpp"});

    auto str_ptr = ov::util::trim_file_name(file_path);
    EXPECT_EQ(file_path, str_ptr);
}

TEST_F(TrimFileTest, absolute_path_to_source_forward_slash_always_supported) {
    const auto exp_path = std::string("src/test_src.cpp");

    const auto file_path = std::string("home/user/") + project_dir_name + "/src/test_src.cpp";
    auto str_ptr = ov::util::trim_file_name(file_path);
    EXPECT_EQ(exp_path, str_ptr);
}

TEST_F(TrimFileTest, relatice_path_to_source_forward_slash_always_supported) {
    const auto exp_path = std::string("src/test_src.cpp");

    const auto file_path = std::string("../../") + project_dir_name + "/src/test_src.cpp";
    auto str_ptr = ov::util::trim_file_name(file_path);
    EXPECT_EQ(exp_path, str_ptr);
}
