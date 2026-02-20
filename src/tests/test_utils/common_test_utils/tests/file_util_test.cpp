// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/util/file_util.hpp"

namespace ov::test {
using std::string;

TEST(file_util, path_join) {
    {
        string s1 = "";
        string s2 = "";

        EXPECT_EQ("", ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "";
        string s2 = "/test1/test2";

        EXPECT_EQ("/test1/test2", ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "";
        string s2 = "/test1/test2/";

        EXPECT_EQ("/test1/test2/", ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "";
        string s2 = "test1/test2";

        EXPECT_EQ("test1/test2", ov::util::path_join({s1, s2}));
    }

    {
        string s1 = "/x1/x2";
        string s2 = "";

        EXPECT_EQ("/x1/x2", ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "/";

        EXPECT_EQ("/", ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "/x1/x2";
        string s2 = "/test1/test2";

        EXPECT_EQ("/test1/test2", ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "test1/test2";

        EXPECT_EQ("/x1/x2/test1/test2", ov::util::path_join({s1, s2}));
    }
    {
        string s1 = "/x1/x2";
        string s2 = "test1/test2";

#ifndef _WIN32
        EXPECT_EQ("/x1/x2/test1/test2", ov::util::path_join({s1, s2}));
#else
        EXPECT_EQ("/x1/x2\\test1/test2", ov::util::path_join({s1, s2}));
#endif
    }
    {
        string s1 = "/";
        string s2 = "test1/test2";

        EXPECT_EQ("/test1/test2", ov::util::path_join({s1, s2}));
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
    const auto exp_path = ov::util::path_join({"src", "test_src.cpp"}).string();

    const auto file_path = ov::util::path_join({"..", "..", "..", project_dir_name, "src", "test_src.cpp"}).string();

    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(exp_path, str_ptr);
}

TEST_F(TrimFileTest, relative_path_to_source_but_no_project_dir) {
    const auto file_path = ov::util::path_join({"..", "..", "..", "src", "test_src.cpp"}).string();

    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(file_path, str_ptr);
}

TEST_F(TrimFileTest, absolute_path_to_source) {
    const auto exp_path = ov::util::path_join({"src", "test_src.cpp"}).string();

    const auto file_path = ov::util::path_join({"home", "user", project_dir_name, "src", "test_src.cpp"}).string();

    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(exp_path, str_ptr);
}

TEST_F(TrimFileTest, absolute_path_to_source_but_no_project_dir) {
    const auto file_path = ov::util::path_join({"home", "user", "src", "test_src.cpp"}).string();

    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(file_path, str_ptr);
}

TEST_F(TrimFileTest, absolute_path_to_source_forward_slash_always_supported) {
    const auto exp_path = std::string("src/test_src.cpp");

    const auto file_path = std::string("home/user/") + project_dir_name + "/src/test_src.cpp";
    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(exp_path, str_ptr);
}

TEST_F(TrimFileTest, relatice_path_to_source_forward_slash_always_supported) {
    const auto exp_path = std::string("src/test_src.cpp");

    const auto file_path = std::string("../../") + project_dir_name + "/src/test_src.cpp";
    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(exp_path, str_ptr);
}

TEST(file_util, path_cast) {
    // from char to char
    EXPECT_STREQ("", std::filesystem::path("").string().c_str());
    EXPECT_STREQ("file.txt", std::filesystem::path("file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", std::filesystem::path("./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", std::filesystem::path("~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", std::filesystem::path("/usr/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("C:\\Users\\file.txt", std::filesystem::path("C:\\Users\\file.txt").string().c_str());

    // from char8_t to char
    EXPECT_STREQ("", std::filesystem::path(u8"").string().c_str());
    EXPECT_STREQ("file.txt", std::filesystem::path(u8"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", std::filesystem::path(u8"./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", std::filesystem::path(u8"~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", std::filesystem::path(u8"/usr/local/file.txt").generic_string().c_str());

    // from char16_t to char
    EXPECT_STREQ("", std::filesystem::path(u"").string().c_str());
    EXPECT_STREQ("file.txt", std::filesystem::path(u"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", std::filesystem::path(u"./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", std::filesystem::path(u"~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", std::filesystem::path(u"/usr/local/file.txt").generic_string().c_str());

    // from char32_t to char
    EXPECT_STREQ("", std::filesystem::path(U"").string().c_str());
    EXPECT_STREQ("file.txt", std::filesystem::path(U"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", std::filesystem::path(U"./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", std::filesystem::path(U"~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", std::filesystem::path(U"/usr/local/file.txt").generic_string().c_str());

    // from char to wchar_t
    EXPECT_STREQ(L"", std::filesystem::path("").wstring().c_str());
    EXPECT_STREQ(L"file.txt", std::filesystem::path("file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", std::filesystem::path("./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", std::filesystem::path("~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", std::filesystem::path("/usr/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"C:\\Users\\file.txt", std::filesystem::path("C:\\Users\\file.txt").wstring().c_str());

    // from char8_t to wchar_t
    EXPECT_STREQ(L"", std::filesystem::path(u8"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", std::filesystem::path(u8"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", std::filesystem::path(u8"./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", std::filesystem::path(u8"~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", std::filesystem::path(u8"/usr/local/file.txt").generic_wstring().c_str());

    // from char16_t to wchar_t
    EXPECT_STREQ(L"", std::filesystem::path(u"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", std::filesystem::path(u"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", std::filesystem::path(u"./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", std::filesystem::path(u"~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", std::filesystem::path(u"/usr/local/file.txt").generic_wstring().c_str());

    // from char32_t to wchar_t
    EXPECT_STREQ(L"", std::filesystem::path(U"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", std::filesystem::path(U"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", std::filesystem::path(U"./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", std::filesystem::path(U"~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", std::filesystem::path(U"/usr/local/file.txt").generic_wstring().c_str());

    // from char to u16string
    EXPECT_TRUE(std::u16string(u"") == std::filesystem::path("").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == std::filesystem::path("file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == std::filesystem::path("./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == std::filesystem::path("~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == std::filesystem::path("/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"C:\\Users\\file.txt") == std::filesystem::path("C:\\Users\\file.txt").u16string());

    // from char8_t to u16string
    EXPECT_TRUE(std::u16string(u"") == std::filesystem::path(u8"").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == std::filesystem::path(u8"file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == std::filesystem::path(u8"./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == std::filesystem::path(u8"~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == std::filesystem::path(u8"/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"C:\\Users\\file.txt") == std::filesystem::path(u8"C:\\Users\\file.txt").u16string());

    // from char16_t to u16string
    EXPECT_TRUE(std::u16string(u"") == std::filesystem::path(u"").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == std::filesystem::path(u"file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == std::filesystem::path(u"./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == std::filesystem::path(u"~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == std::filesystem::path(u"/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"C:\\Users\\file.txt") == std::filesystem::path(u"C:\\Users\\file.txt").u16string());

    // from char32_t to u16string
    EXPECT_TRUE(std::u16string(u"") == std::filesystem::path(U"").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == std::filesystem::path(U"file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == std::filesystem::path(U"./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == std::filesystem::path(U"~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == std::filesystem::path(U"/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"C:\\Users\\file.txt") == std::filesystem::path(U"C:\\Users\\file.txt").u16string());
}

TEST(file_util, path_cast_unicode) {
    EXPECT_EQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗1.txt", std::filesystem::path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗1.txt").generic_string());
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗6.txt") ==
                std::filesystem::path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗6.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗7.txt") ==
                std::filesystem::path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗7.txt").generic_u16string());

#if !defined(_MSC_VER) && defined(OPENVINO_CPP_VER_AT_LEAST_20)
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗9.txt") ==
                std::filesystem::path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗9.txt").generic_u8string());
#endif
#if defined(OPENVINO_CPP_VER_AT_LEAST_20)
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗10.txt") ==
                std::filesystem::path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗10.txt").generic_u8string());
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗11.txt") ==
                std::filesystem::path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗11.txt").generic_u8string());
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗12.txt") ==
                std::filesystem::path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗12.txt").generic_u8string());
#else
    EXPECT_EQ(std::string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗14.txt"),
              std::filesystem::path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗14.txt").generic_u8string());
    EXPECT_EQ(std::string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗15.txt"),
              std::filesystem::path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗15.txt").generic_u8string());
#endif

#if !defined(_MSC_VER) || defined(_MSC_VER) && defined(OPENVINO_CPP_VER_AT_LEAST_20)
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗8.txt") ==
                std::filesystem::path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗8.txt").generic_u16string());
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗18.txt") ==
                std::filesystem::path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗18.txt").u32string());
#endif

#if defined(OPENVINO_CPP_VER_AT_LEAST_20) && \
    (defined(_MSC_VER) ||                    \
     !defined(_MSC_VER) && defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17))
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗16.txt") ==
                std::filesystem::path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗16.txt").generic_u8string());
    EXPECT_TRUE(std::wstring(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗29.txt") ==
                std::filesystem::path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗29.txt").wstring());
#endif
}

#if !defined(_MSC_VER)
TEST(file_util, path_cast_unicode_from_string) {
    EXPECT_TRUE(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗13.txt" ==
                std::filesystem::path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗13.txt").generic_u8string());
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗5.txt") ==
                std::filesystem::path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗5.txt").generic_u16string());
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗21.txt") ==
                std::filesystem::path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗21.txt").u32string());
}

TEST(file_util, path_cast_unicode_to_string) {
    EXPECT_EQ(std::string("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗4.txt"),
              std::filesystem::path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗4.txt").generic_string());
    EXPECT_EQ(std::string("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗2.txt"),
              std::filesystem::path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗2.txt").generic_string());
    EXPECT_EQ(std::string("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗3.txt"),
              std::filesystem::path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗3.txt").generic_string());
}
#endif

#if !defined(_MSC_VER) && defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17)
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95048
// https://stackoverflow.com/questions/58521857/cross-platform-way-to-handle-stdstring-stdwstring-with-stdfilesystempath

TEST(file_util, path_cast_unicode_from_string_to_wstring) {
    EXPECT_TRUE(std::wstring(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗27.txt") ==
                std::filesystem::path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗27.txt").generic_wstring());
}
TEST(file_util, path_cast_unicode_from_wstring_to_string) {
    EXPECT_STREQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗25.txt",
                 std::filesystem::path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗25.txt").generic_string().c_str());
}
#endif

TEST(file_util, path_cast_to_u32string) {
    // from char to u32string
    EXPECT_TRUE(std::u32string(U"") == std::filesystem::path("").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == std::filesystem::path("file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == std::filesystem::path("./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == std::filesystem::path("~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == std::filesystem::path("/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"C:\\Users\\file.txt") == std::filesystem::path("C:\\Users\\file.txt").u32string());

    // from char8_t to u32string
    EXPECT_TRUE(std::u32string(U"") == std::filesystem::path(u8"").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == std::filesystem::path(u8"file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == std::filesystem::path(u8"./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == std::filesystem::path(u8"~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == std::filesystem::path(u8"/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"C:\\Users\\file.txt") == std::filesystem::path(u8"C:\\Users\\file.txt").u32string());

    // from char16_t to u32string
    EXPECT_TRUE(std::u32string(U"") == std::filesystem::path(u"").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == std::filesystem::path(u"file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == std::filesystem::path(u"./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == std::filesystem::path(u"~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == std::filesystem::path(u"/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"C:\\Users\\file.txt") == std::filesystem::path(u"C:\\Users\\file.txt").u32string());

    // from char32_t to u32string
    EXPECT_TRUE(std::u32string(U"") == std::filesystem::path(U"").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == std::filesystem::path(U"file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == std::filesystem::path(U"./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == std::filesystem::path(U"~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == std::filesystem::path(U"/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"C:\\Users\\file.txt") == std::filesystem::path(U"C:\\Users\\file.txt").u32string());

    // from char16_t, char32_t to u32string
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗23.txt") ==
                std::filesystem::path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗23.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗24.txt") ==
                std::filesystem::path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗24.txt").u32string());
}

#if defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17)
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95048
// https://stackoverflow.com/questions/58521857/cross-platform-way-to-handle-stdstring-stdwstring-with-stdfilesystempath

TEST(file_util, path_cast_from_wstring) {
    // from wchar_t to char
    EXPECT_STREQ("", std::filesystem::path(L"").string().c_str());
    EXPECT_STREQ("file.txt", std::filesystem::path(L"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", std::filesystem::path(L"./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", std::filesystem::path(L"~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", std::filesystem::path(L"/usr/local/file.txt").generic_string().c_str());

    // from wchar_t to wchar_t
    EXPECT_STREQ(L"", std::filesystem::path(L"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", std::filesystem::path(L"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", std::filesystem::path(L"./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", std::filesystem::path(L"~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", std::filesystem::path(L"/usr/local/file.txt").generic_wstring().c_str());

    // from wchar_t to char16_t
    EXPECT_TRUE(std::u16string(u"") == std::filesystem::path(L"").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == std::filesystem::path(L"file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == std::filesystem::path(L"./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == std::filesystem::path(L"~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == std::filesystem::path(L"/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗26.txt") ==
                std::filesystem::path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗26.txt").generic_u16string());
}

TEST(file_util, path_cast_to_wstring) {
    // from char16_t, char32_t to wchar_t
    EXPECT_STREQ(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗28.txt",
                 std::filesystem::path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗28.txt").generic_wstring().c_str());

    EXPECT_STREQ(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗30.txt",
                 std::filesystem::path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗30.txt").wstring().c_str());
}

TEST(file_util, path_cast_from_wstring_to_u32string) {
    EXPECT_TRUE(std::u32string(U"") == std::filesystem::path(L"").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == std::filesystem::path(L"file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == std::filesystem::path(L"./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == std::filesystem::path(L"~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == std::filesystem::path(L"/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗31.txt") ==
                std::filesystem::path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗31.txt").u32string());
}
#endif

TEST(file_util, path_cast_from_wstring_to_char8_t) {
    // from wchar_t to char8_t
#if defined(OPENVINO_CPP_VER_AT_LEAST_20)
    EXPECT_TRUE(std::u8string(u8"") == std::filesystem::path(L"").u8string());
    EXPECT_TRUE(std::u8string(u8"file.txt") == std::filesystem::path(L"file.txt").u8string());
    EXPECT_TRUE(std::u8string(u8"./local/file.txt") == std::filesystem::path(L"./local/file.txt").generic_u8string());
    EXPECT_TRUE(std::u8string(u8"~/local/file.txt") == std::filesystem::path(L"~/local/file.txt").generic_u8string());
    EXPECT_TRUE(std::u8string(u8"/usr/local/file.txt") == std::filesystem::path(L"/usr/local/file.txt").generic_u8string());
#elif defined(OPENVINO_CPP_VER_AT_LEAST_17)
    EXPECT_EQ(std::string(""), std::filesystem::path(L"").u8string());
    EXPECT_EQ(std::string("file.txt"), std::filesystem::path(L"file.txt").u8string());
    EXPECT_EQ(std::string("./local/file.txt"), std::filesystem::path(L"./local/file.txt").generic_u8string());
    EXPECT_EQ(std::string("~/local/file.txt"), std::filesystem::path(L"~/local/file.txt").generic_u8string());
    EXPECT_EQ(std::string("/usr/local/file.txt"), std::filesystem::path(L"/usr/local/file.txt").generic_u8string());
#endif
}

TEST(file_util, unicode_path_cast_from_wstring_to_char8_t) {
    // from wchar_t to char8_t
#if defined(OPENVINO_CPP_VER_AT_LEAST_20) && defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && \
    defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17)
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗32.txt") ==
                std::filesystem::path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗32.txt").generic_u8string());

#elif defined(OPENVINO_CPP_VER_AT_LEAST_17) && defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && \
    defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17)
    EXPECT_EQ(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗33.txt",
              std::filesystem::path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗33.txt").generic_u8string());
#endif
}

class FileUtilTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary files for testing
        { std::ofstream outfile("test_file_0.txt"); }
        {
            std::ofstream outfile("test_file_20.txt");
            outfile << "This is a test file.";
        }
        {
            std::ofstream outfile("test_file_20x1000.txt");
            for (int i = 0; i < 1000; ++i) {
                outfile << "This is a test file.";
            }
        }
        {
            std::ofstream outfile("test_file_raw_bytes_746.txt", std::ios::binary);
            std::vector<char> buffer(746, 0);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 255);

            std::generate(buffer.begin(), buffer.end(), [&]() {
                return static_cast<char>(dis(gen));
            });

            outfile.write(buffer.data(), buffer.size());
        }

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef OPENVINO_CPP_VER_AT_LEAST_20
        {
            std::ofstream outfile(std::filesystem::path(u8"这是_u8.txt"));
            outfile << "This is a test file.";
        }
#    endif
        {
            std::ofstream outfile(std::filesystem::path(u"这是_u16.txt"));
            outfile << "This is a test file.";
        }
        {
            std::ofstream outfile(std::filesystem::path(U"这是_u32.txt"));
            outfile << "This is a test file.";
        }
        {
            std::ofstream outfile(ov::util::make_path(L"这是_wchar.txt"));
            outfile << "This is a test file.";
        }
#endif

#if defined(__ANDROID__) || defined(ANDROID)
        {
            std::ofstream outfile("android_test_file_20.txt");
            outfile << "This is a test file.";
        }
#endif
    }

    void TearDown() override {
        // Remove the temporary files after testing
        std::filesystem::remove("test_file_0.txt");
        std::filesystem::remove("test_file_20.txt");
        std::filesystem::remove("test_file_raw_bytes_746.txt");
        std::filesystem::remove("test_file_20x1000.txt");
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
        std::filesystem::remove(u8"这是_u8.txt");
        std::filesystem::remove(u"这是_u16.txt");
        std::filesystem::remove(U"这是_u32.txt");
        std::filesystem::remove(ov::util::make_path(L"这是_wchar.txt"));
#endif
#if defined(__ANDROID__) || defined(ANDROID)
        std::filesystem::remove("android_test_file_20.txt");
#endif
    }
};

TEST_F(FileUtilTest, FileSizeNonExistentFileTest) {
    EXPECT_EQ(ov::util::file_size("non_existent_file.txt"), -1);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path("non_existent_file.txt")), -1);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"non_existent_file.txt")), -1);
}

TEST_F(FileUtilTest, EmptyFileSizeTest) {
#ifdef OPENVINO_CPP_VER_AT_LEAST_20
    EXPECT_EQ(ov::util::file_size(u8"test_file_0.txt"), 0);
#endif
    EXPECT_EQ(ov::util::file_size("test_file_0.txt"), 0);
    EXPECT_EQ(ov::util::file_size(u"test_file_0.txt"), 0);
    EXPECT_EQ(ov::util::file_size(U"test_file_0.txt"), 0);
    EXPECT_EQ(ov::util::file_size(std::wstring(L"test_file_0.txt")), 0);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path("test_file_0.txt")), 0);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(u8"test_file_0.txt")), 0);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(u"test_file_0.txt")), 0);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(U"test_file_0.txt")), 0);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(L"test_file_0.txt")), 0);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"test_file_0.txt")), 0);
}

TEST_F(FileUtilTest, FileSizeTest) {
    EXPECT_EQ(ov::util::file_size("test_file_20.txt"), 20);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"test_file_20.txt")), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path("test_file_20.txt")), 20);
}

TEST_F(FileUtilTest, FileSizeRawBytesTest) {
    EXPECT_EQ(ov::util::file_size("test_file_raw_bytes_746.txt"), 746);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"test_file_raw_bytes_746.txt")), 746);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path("test_file_raw_bytes_746.txt")), 746);
}

TEST_F(FileUtilTest, LargeFileSizeTest) {
    EXPECT_EQ(ov::util::file_size("test_file_20x1000.txt"), 20 * 1000);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"test_file_20x1000.txt")), 20 * 1000);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path("test_file_20x1000.txt")), 20 * 1000);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

#    ifdef OPENVINO_CPP_VER_AT_LEAST_20
TEST_F(FileUtilTest, u8FileSizeTest) {
    EXPECT_EQ(ov::util::file_size(u8"这是_u8.txt"), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(u8"这是_u8.txt")), 20);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"这是_u8.txt")), 20);
}
#    endif

TEST_F(FileUtilTest, u16FileSizeTest) {
    EXPECT_EQ(ov::util::file_size("这是_u16.txt"), 20);
#    ifdef OPENVINO_CPP_VER_AT_LEAST_20
    EXPECT_EQ(ov::util::file_size(u8"这是_u16.txt"), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(u8"这是_u16.txt")), 20);
#    endif
    EXPECT_EQ(ov::util::file_size(u"这是_u16.txt"), 20);
    EXPECT_EQ(ov::util::file_size(U"这是_u16.txt"), 20);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"这是_u16.txt")), 20);
    EXPECT_EQ(ov::util::file_size(std::wstring(L"这是_u16.txt")), 20);

    EXPECT_EQ(ov::util::file_size(std::filesystem::path(u"这是_u16.txt")), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(U"这是_u16.txt")), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(std::wstring(L"这是_u16.txt"))), 20);
}

TEST_F(FileUtilTest, u32FileSizeTest) {
    EXPECT_EQ(ov::util::file_size(U"这是_u32.txt"), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(U"这是_u32.txt")), 20);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"这是_u32.txt")), 20);
}

TEST_F(FileUtilTest, wcharFileSizeTest) {
#    ifdef _MSC_VER
    EXPECT_EQ(ov::util::file_size(L"这是_wchar.txt"), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path(L"这是_wchar.txt")), 20);
#    else
    EXPECT_EQ(ov::util::file_size("这是_wchar.txt"), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path("这是_wchar.txt")), 20);
#    endif
    EXPECT_EQ(ov::util::file_size("这是_wchar.txt"), 20);
    EXPECT_EQ(ov::util::file_size(ov::util::make_path(L"这是_wchar.txt")), 20);
}
#endif

#if defined(__ANDROID__) || defined(ANDROID)
TEST_F(FileUtilTest, androidFileSizeTest) {
    EXPECT_EQ(ov::util::file_size("android_test_file_20.txt"), 20);
    EXPECT_EQ(ov::util::file_size(L"android_test_file_20.txt"), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path("android_test_file_20.txt")), 20);
}
TEST_F(FileUtilTest, androidWithCutFileSizeTest) {
    EXPECT_EQ(ov::util::file_size("android_test_file_20.txt!_to_cut.jar"), 20);
    EXPECT_EQ(ov::util::file_size(L"android_test_file_20.txt!_to_cut.jar"), 20);
    EXPECT_EQ(ov::util::file_size(std::filesystem::path("android_test_file_20.txt!_to_cut.jar")), 20);
}
#endif

using FileUtilTestP = UnicodePathTest;

INSTANTIATE_TEST_SUITE_P(string_paths,
                         FileUtilTestP,
                         testing::Values("test_encoder/test_encoder.encrypted/",
                                         "test_encoder/test_encoder.encrypted"));
INSTANTIATE_TEST_SUITE_P(u16_paths,
                         FileUtilTestP,
                         testing::Values(u"test_encoder/dot.folder", u"test_encoder/dot.folder/"));

INSTANTIATE_TEST_SUITE_P(u32_paths,
                         FileUtilTestP,
                         testing::Values(U"test_encoder/dot.folder", U"test_encoder/dot.folder/"));

INSTANTIATE_TEST_SUITE_P(wstring_paths,
                         FileUtilTestP,
                         testing::Values(L"test_encoder/test_encoder.encrypted",
                                         L"test_encoder/test_encoder.encrypted/"));

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
INSTANTIATE_TEST_SUITE_P(
    unicode_paths,
    FileUtilTestP,
    testing::Values("这是.folder", L"这是_folder", L"这是_folder/", u"这是_folder/", U"这是_folder/"));
#endif

TEST_P(FileUtilTestP, create_directories) {
    const auto test_dir = utils::generateTestFilePrefix();
    const auto path = std::filesystem::path(test_dir) / get_path_param();
    const auto exp_path = std::filesystem::path(test_dir) / utils::to_fs_path(GetParam());

    ov::util::create_directory_recursive(path);

    EXPECT_EQ(path, exp_path);
    ASSERT_TRUE(std::filesystem::exists(path));
    ASSERT_TRUE(std::filesystem::exists(exp_path));

    std::filesystem::remove_all(test_dir);
}
}  // namespace ov::test
