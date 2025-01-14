// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_util.hpp"

#include <gtest/gtest.h>

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/util/file_path.hpp"

using namespace std;
using namespace ov;

TEST(file_util, path_join) {
    {
        string s1 = "";
        string s2 = "";

        EXPECT_STREQ("", ov::util::path_join({s1, s2}).c_str());
    }
    {
        string s1 = "";
        string s2 = "/test1/test2";

        EXPECT_STREQ("/test1/test2", ov::util::path_join({s1, s2}).c_str());
    }
    {
        string s1 = "";
        string s2 = "/test1/test2/";

        EXPECT_STREQ("/test1/test2/", ov::util::path_join({s1, s2}).c_str());
    }
    {
        string s1 = "";
        string s2 = "test1/test2";

        EXPECT_STREQ("test1/test2", ov::util::path_join({s1, s2}).c_str());
    }

    {
        string s1 = "/x1/x2";
        string s2 = "";

        EXPECT_STREQ("/x1/x2", ov::util::path_join({s1, s2}).c_str());
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "/";

        EXPECT_STREQ("/", ov::util::path_join({s1, s2}).c_str());
    }
    {
        string s1 = "/x1/x2";
        string s2 = "/test1/test2";

        EXPECT_STREQ("/test1/test2", ov::util::path_join({s1, s2}).c_str());
    }
    {
        string s1 = "/x1/x2/";
        string s2 = "test1/test2";

        EXPECT_STREQ("/x1/x2/test1/test2", ov::util::path_join({s1, s2}).c_str());
    }
    {
        string s1 = "/x1/x2";
        string s2 = "test1/test2";

#ifndef _WIN32
        EXPECT_STREQ("/x1/x2/test1/test2", ov::util::path_join({s1, s2}).c_str());
#else
        EXPECT_STREQ("/x1/x2\\test1/test2", ov::util::path_join({s1, s2}).c_str());
#endif
    }
    {
        string s1 = "/";
        string s2 = "test1/test2";

        EXPECT_STREQ("/test1/test2", ov::util::path_join({s1, s2}).c_str());
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

    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(exp_path, str_ptr);
}

TEST_F(TrimFileTest, relative_path_to_source_but_no_project_dir) {
    const auto file_path = ov::util::path_join({"..", "..", "..", "src", "test_src.cpp"});

    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(file_path, str_ptr);
}

TEST_F(TrimFileTest, absolute_path_to_source) {
    const auto exp_path = ov::util::path_join({"src", "test_src.cpp"});

    const auto file_path = ov::util::path_join({"home", "user", project_dir_name, "src", "test_src.cpp"});

    auto str_ptr = ov::util::trim_file_name(file_path.c_str());
    EXPECT_EQ(exp_path, str_ptr);
}

TEST_F(TrimFileTest, absolute_path_to_source_but_no_project_dir) {
    const auto file_path = ov::util::path_join({"home", "user", "src", "test_src.cpp"});

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
    EXPECT_STREQ("", ov::util::Path("").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path("file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path("./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path("~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path("/usr/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("C:\\Users\\file.txt", ov::util::Path("C:\\Users\\file.txt").string().c_str());

    // from char8_t to char
    EXPECT_STREQ("", ov::util::Path(u8"").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(u8"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(u8"./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(u8"~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(u8"/usr/local/file.txt").generic_string().c_str());

    // from char16_t to char
    EXPECT_STREQ("", ov::util::Path(u"").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(u"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(u"./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(u"~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(u"/usr/local/file.txt").generic_string().c_str());

    // from char32_t to char
    EXPECT_STREQ("", ov::util::Path(U"").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(U"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(U"./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(U"~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(U"/usr/local/file.txt").generic_string().c_str());

    // from char to wchar_t
    EXPECT_STREQ(L"", ov::util::Path("").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path("file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path("./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path("~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path("/usr/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"C:\\Users\\file.txt", ov::util::Path("C:\\Users\\file.txt").wstring().c_str());

    // from char8_t to wchar_t
    EXPECT_STREQ(L"", ov::util::Path(u8"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path(u8"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path(u8"./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path(u8"~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path(u8"/usr/local/file.txt").generic_wstring().c_str());

    // from char16_t to wchar_t
    EXPECT_STREQ(L"", ov::util::Path(u"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path(u"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path(u"./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path(u"~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path(u"/usr/local/file.txt").generic_wstring().c_str());

    // from char32_t to wchar_t
    EXPECT_STREQ(L"", ov::util::Path(U"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path(U"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path(U"./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path(U"~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path(U"/usr/local/file.txt").generic_wstring().c_str());

    // from char to u16string
    EXPECT_TRUE(std::u16string(u"") == ov::util::Path("").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == ov::util::Path("file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == ov::util::Path("./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == ov::util::Path("~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == ov::util::Path("/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"C:\\Users\\file.txt") == ov::util::Path("C:\\Users\\file.txt").u16string());

    // from char8_t to u16string
    EXPECT_TRUE(std::u16string(u"") == ov::util::Path(u8"").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == ov::util::Path(u8"file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == ov::util::Path(u8"./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == ov::util::Path(u8"~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == ov::util::Path(u8"/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"C:\\Users\\file.txt") == ov::util::Path(u8"C:\\Users\\file.txt").u16string());

    // from char16_t to u16string
    EXPECT_TRUE(std::u16string(u"") == ov::util::Path(u"").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == ov::util::Path(u"file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == ov::util::Path(u"./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == ov::util::Path(u"~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == ov::util::Path(u"/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"C:\\Users\\file.txt") == ov::util::Path(u"C:\\Users\\file.txt").u16string());

    // from char32_t to u16string
    EXPECT_TRUE(std::u16string(u"") == ov::util::Path(U"").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == ov::util::Path(U"file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == ov::util::Path(U"./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == ov::util::Path(U"~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == ov::util::Path(U"/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"C:\\Users\\file.txt") == ov::util::Path(U"C:\\Users\\file.txt").u16string());
}

// There are known issues related with usage of std::filesystem::path unocode represenataion:
// https://jira.devtools.intel.com/browse/CVS-160477
// * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95048
// * https://stackoverflow.com/questions/58521857/cross-platform-way-to-handle-stdstring-stdwstring-with-stdfilesystempath
#if !defined(__GNUC__) || (__GNUC__ > 12 || __GNUC__ == 12 && __GNUC_MINOR__ >= 3)
#    define GCC_NOT_USED_OR_VER_AT_LEAST_12_3
#endif

#if !defined(__clang__) || defined(__clang__) && __clang_major__ >= 17
#    define CLANG_NOT_USED_OR_VER_AT_LEAST_17
#endif

TEST(file_util, path_cast_unicode) {
    EXPECT_EQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗1.txt", ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗1.txt").generic_string());
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗6.txt") ==
                ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗6.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗7.txt") ==
                ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗7.txt").generic_u16string());

#if !defined(_MSC_VER) && defined(OPENVINO_CPP_VER_AT_LEAST_20)
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗9.txt") ==
                ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗9.txt").generic_u8string());
#endif
#if defined(OPENVINO_CPP_VER_AT_LEAST_20)
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗10.txt") ==
                ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗10.txt").generic_u8string());
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗11.txt") ==
                ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗11.txt").generic_u8string());
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗12.txt") ==
                ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗12.txt").generic_u8string());
#elif defined(OPENVINO_CPP_VER_AT_LEAST_17)
    EXPECT_EQ(std::string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗14.txt"),
              ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗14.txt").generic_u8string());
    EXPECT_EQ(std::string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗15.txt"),
              ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗15.txt").generic_u8string());
#endif

#if !defined(_MSC_VER) || defined(_MSC_VER) && defined(OPENVINO_CPP_VER_AT_LEAST_20)
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗8.txt") ==
                ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗8.txt").generic_u16string());
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗18.txt") ==
                ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗18.txt").u32string());
#endif

#if defined(OPENVINO_CPP_VER_AT_LEAST_20) && \
    (defined(_MSC_VER) ||                    \
     !defined(_MSC_VER) && defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17))
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗16.txt") ==
                ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗16.txt").generic_u8string());
    EXPECT_TRUE(std::wstring(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗29.txt") ==
                ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗29.txt").wstring());
#endif
}

#if !defined(_MSC_VER)
TEST(file_util, path_cast_unicode_from_string) {
    EXPECT_TRUE(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗13.txt" ==
                ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗13.txt").generic_u8string());
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗5.txt") ==
                ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗5.txt").generic_u16string());
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗21.txt") ==
                ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗21.txt").u32string());
}

TEST(file_util, path_cast_unicode_to_string) {
    EXPECT_EQ(std::string("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗4.txt"),
              ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗4.txt").generic_string());
    EXPECT_EQ(std::string("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗2.txt"),
              ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗2.txt").generic_string());
    EXPECT_EQ(std::string("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗3.txt"),
              ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗3.txt").generic_string());
}
#endif

#if !defined(_MSC_VER) && defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17)
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95048
// https://stackoverflow.com/questions/58521857/cross-platform-way-to-handle-stdstring-stdwstring-with-stdfilesystempath

TEST(file_util, path_cast_unicode_from_string_to_wstring) {
    EXPECT_TRUE(std::wstring(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗27.txt") ==
                ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗27.txt").generic_wstring());
}
TEST(file_util, path_cast_unicode_from_wstring_to_string) {
    EXPECT_STREQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗25.txt",
                 ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗25.txt").generic_string().c_str());
}
#endif

TEST(file_util, path_cast_to_u32string) {
    // from char to u32string
    EXPECT_TRUE(std::u32string(U"") == ov::util::Path("").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == ov::util::Path("file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == ov::util::Path("./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == ov::util::Path("~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == ov::util::Path("/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"C:\\Users\\file.txt") == ov::util::Path("C:\\Users\\file.txt").u32string());

    // from char8_t to u32string
    EXPECT_TRUE(std::u32string(U"") == ov::util::Path(u8"").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == ov::util::Path(u8"file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == ov::util::Path(u8"./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == ov::util::Path(u8"~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == ov::util::Path(u8"/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"C:\\Users\\file.txt") == ov::util::Path(u8"C:\\Users\\file.txt").u32string());

    // from char16_t to u32string
    EXPECT_TRUE(std::u32string(U"") == ov::util::Path(u"").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == ov::util::Path(u"file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == ov::util::Path(u"./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == ov::util::Path(u"~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == ov::util::Path(u"/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"C:\\Users\\file.txt") == ov::util::Path(u"C:\\Users\\file.txt").u32string());

    // from char32_t to u32string
    EXPECT_TRUE(std::u32string(U"") == ov::util::Path(U"").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == ov::util::Path(U"file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == ov::util::Path(U"./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == ov::util::Path(U"~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == ov::util::Path(U"/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"C:\\Users\\file.txt") == ov::util::Path(U"C:\\Users\\file.txt").u32string());

    // from char16_t, char32_t to u32string
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗23.txt") ==
                ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗23.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗24.txt") ==
                ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗24.txt").u32string());
}

#if defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17)
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95048
// https://stackoverflow.com/questions/58521857/cross-platform-way-to-handle-stdstring-stdwstring-with-stdfilesystempath

TEST(file_util, path_cast_from_wstring) {
    // from wchar_t to char
    EXPECT_STREQ("", ov::util::Path(L"").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(L"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(L"./local/file.txt").generic_string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(L"~/local/file.txt").generic_string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(L"/usr/local/file.txt").generic_string().c_str());

    // from wchar_t to wchar_t
    EXPECT_STREQ(L"", ov::util::Path(L"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path(L"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path(L"./local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path(L"~/local/file.txt").generic_wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path(L"/usr/local/file.txt").generic_wstring().c_str());

    // from wchar_t to char16_t
    EXPECT_TRUE(std::u16string(u"") == ov::util::Path(L"").u16string());
    EXPECT_TRUE(std::u16string(u"file.txt") == ov::util::Path(L"file.txt").u16string());
    EXPECT_TRUE(std::u16string(u"./local/file.txt") == ov::util::Path(L"./local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/local/file.txt") == ov::util::Path(L"~/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"/usr/local/file.txt") == ov::util::Path(L"/usr/local/file.txt").generic_u16string());
    EXPECT_TRUE(std::u16string(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗26.txt") ==
                ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗26.txt").generic_u16string());
}

TEST(file_util, path_cast_to_wstring) {
    // from char16_t, char32_t to wchar_t
    EXPECT_STREQ(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗28.txt",
                 ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗28.txt").generic_wstring().c_str());

    EXPECT_STREQ(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗30.txt",
                 ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗30.txt").wstring().c_str());
}

TEST(file_util, path_cast_from_wstring_to_u32string) {
    EXPECT_TRUE(std::u32string(U"") == ov::util::Path(L"").u32string());
    EXPECT_TRUE(std::u32string(U"file.txt") == ov::util::Path(L"file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"./local/file.txt") == ov::util::Path(L"./local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/local/file.txt") == ov::util::Path(L"~/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"/usr/local/file.txt") == ov::util::Path(L"/usr/local/file.txt").u32string());
    EXPECT_TRUE(std::u32string(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗31.txt") ==
                ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗31.txt").u32string());
}
#endif

TEST(file_util, path_cast_from_wstring_to_char8_t) {
    // from wchar_t to char8_t
#if defined(OPENVINO_CPP_VER_AT_LEAST_20)
    EXPECT_TRUE(std::u8string(u8"") == ov::util::Path(L"").u8string());
    EXPECT_TRUE(std::u8string(u8"file.txt") == ov::util::Path(L"file.txt").u8string());
    EXPECT_TRUE(std::u8string(u8"./local/file.txt") == ov::util::Path(L"./local/file.txt").generic_u8string());
    EXPECT_TRUE(std::u8string(u8"~/local/file.txt") == ov::util::Path(L"~/local/file.txt").generic_u8string());
    EXPECT_TRUE(std::u8string(u8"/usr/local/file.txt") == ov::util::Path(L"/usr/local/file.txt").generic_u8string());
#elif defined(OPENVINO_CPP_VER_AT_LEAST_17)
    EXPECT_EQ(std::string(""), ov::util::Path(L"").u8string());
    EXPECT_EQ(std::string("file.txt"), ov::util::Path(L"file.txt").u8string());
    EXPECT_EQ(std::string("./local/file.txt"), ov::util::Path(L"./local/file.txt").generic_u8string());
    EXPECT_EQ(std::string("~/local/file.txt"), ov::util::Path(L"~/local/file.txt").generic_u8string());
    EXPECT_EQ(std::string("/usr/local/file.txt"), ov::util::Path(L"/usr/local/file.txt").generic_u8string());
#endif
}

TEST(file_util, unicode_path_cast_from_wstring_to_char8_t) {
    // from wchar_t to char8_t
#if defined(OPENVINO_CPP_VER_AT_LEAST_20) && defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && \
    defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17)
    EXPECT_TRUE(std::u8string(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗32.txt") ==
                ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗32.txt").generic_u8string());

#elif defined(OPENVINO_CPP_VER_AT_LEAST_17) && defined(GCC_NOT_USED_OR_VER_AT_LEAST_12_3) && \
    defined(CLANG_NOT_USED_OR_VER_AT_LEAST_17)
    EXPECT_EQ(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗33.txt",
              ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗33.txt").generic_u8string());
#endif
}
