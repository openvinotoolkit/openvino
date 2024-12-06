// Copyright (C) 2018-2024 Intel Corporation
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
    EXPECT_STREQ("./local/file.txt", ov::util::Path("./local/file.txt").string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path("~/local/file.txt").string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path("/usr/local/file.txt").string().c_str());
    EXPECT_STREQ("C:\\Users\\file.txt", ov::util::Path("C:\\Users\\file.txt").string().c_str());
    EXPECT_STREQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").string().c_str());

    // from char8_t to char
    EXPECT_STREQ("", ov::util::Path(u8"").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(u8"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(u8"./local/file.txt").string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(u8"~/local/file.txt").string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(u8"/usr/local/file.txt").string().c_str());
    EXPECT_STREQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").string().c_str());

    // from char16_t to char
    EXPECT_STREQ("", ov::util::Path(u"").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(u"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(u"./local/file.txt").string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(u"~/local/file.txt").string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(u"/usr/local/file.txt").string().c_str());
    EXPECT_STREQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").string().c_str());

    // from char32_t to char
    EXPECT_STREQ("", ov::util::Path(U"").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(U"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(U"./local/file.txt").string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(U"~/local/file.txt").string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(U"/usr/local/file.txt").string().c_str());
    EXPECT_STREQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").string().c_str());

    // from char to wchar_t
    EXPECT_STREQ(L"", ov::util::Path("").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path("file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path("./local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path("~/local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path("/usr/local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"C:\\Users\\file.txt", ov::util::Path("C:\\Users\\file.txt").wstring().c_str());

    // from char8_t to wchar_t
    EXPECT_STREQ(L"", ov::util::Path(u8"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path(u8"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path(u8"./local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path(u8"~/local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path(u8"/usr/local/file.txt").wstring().c_str());

    // from char16_t to wchar_t
    EXPECT_STREQ(L"", ov::util::Path(u"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path(u"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path(u"./local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path(u"~/local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path(u"/usr/local/file.txt").wstring().c_str());

    // from char32_t to wchar_t
    EXPECT_STREQ(L"", ov::util::Path(U"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path(U"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path(U"./local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path(U"~/local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path(U"/usr/local/file.txt").wstring().c_str());

    // from char to u16string
    EXPECT_EQ(u"", ov::util::Path("").u16string());
    EXPECT_EQ(u"file.txt", ov::util::Path("file.txt").u16string());
    EXPECT_EQ(u"./local/file.txt", ov::util::Path("./local/file.txt").u16string());
    EXPECT_EQ(u"~/local/file.txt", ov::util::Path("~/local/file.txt").u16string());
    EXPECT_EQ(u"/usr/local/file.txt", ov::util::Path("/usr/local/file.txt").u16string());
    EXPECT_EQ(u"C:\\Users\\file.txt", ov::util::Path("C:\\Users\\file.txt").u16string());
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());

    // from char8_t to u16string
    EXPECT_EQ(u"", ov::util::Path(u8"").u16string());
    EXPECT_EQ(u"file.txt", ov::util::Path(u8"file.txt").u16string());
    EXPECT_EQ(u"./local/file.txt", ov::util::Path(u8"./local/file.txt").u16string());
    EXPECT_EQ(u"~/local/file.txt", ov::util::Path(u8"~/local/file.txt").u16string());
    EXPECT_EQ(u"/usr/local/file.txt", ov::util::Path(u8"/usr/local/file.txt").u16string());
    EXPECT_EQ(u"C:\\Users\\file.txt", ov::util::Path(u8"C:\\Users\\file.txt").u16string());
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());

    // from char16_t to u16string
    EXPECT_EQ(u"", ov::util::Path(u"").u16string());
    EXPECT_EQ(u"file.txt", ov::util::Path(u"file.txt").u16string());
    EXPECT_EQ(u"./local/file.txt", ov::util::Path(u"./local/file.txt").u16string());
    EXPECT_EQ(u"~/local/file.txt", ov::util::Path(u"~/local/file.txt").u16string());
    EXPECT_EQ(u"/usr/local/file.txt", ov::util::Path(u"/usr/local/file.txt").u16string());
    EXPECT_EQ(u"C:\\Users\\file.txt", ov::util::Path(u"C:\\Users\\file.txt").u16string());
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());

    // from char32_t to u16string
    EXPECT_EQ(u"", ov::util::Path(U"").u16string());
    EXPECT_EQ(u"file.txt", ov::util::Path(U"file.txt").u16string());
    EXPECT_EQ(u"./local/file.txt", ov::util::Path(U"./local/file.txt").u16string());
    EXPECT_EQ(u"~/local/file.txt", ov::util::Path(U"~/local/file.txt").u16string());
    EXPECT_EQ(u"/usr/local/file.txt", ov::util::Path(U"/usr/local/file.txt").u16string());
    EXPECT_EQ(u"C:\\Users\\file.txt", ov::util::Path(U"C:\\Users\\file.txt").u16string());
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());

    // from char, char8_t, char16_t, char32_t to u16string
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());

#ifndef MSVC
    // error C2280: 'std::u32string std::experimental::filesystem::v1::path::u32string(void) const': attempting to
    // reference a deleted function

    // from char to u32string
    EXPECT_EQ(U"", ov::util::Path("").u32string());
    EXPECT_EQ(U"file.txt", ov::util::Path("file.txt").u32string());
    EXPECT_EQ(U"./local/file.txt", ov::util::Path("./local/file.txt").u32string());
    EXPECT_EQ(U"~/local/file.txt", ov::util::Path("~/local/file.txt").u32string());
    EXPECT_EQ(U"/usr/local/file.txt", ov::util::Path("/usr/local/file.txt").u32string());
    EXPECT_EQ(U"C:\\Users\\file.txt", ov::util::Path("C:\\Users\\file.txt").u32string());
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());

    // from char8_t to u32string
    EXPECT_EQ(U"", ov::util::Path(u8"").u32string());
    EXPECT_EQ(U"file.txt", ov::util::Path(u8"file.txt").u32string());
    EXPECT_EQ(U"./local/file.txt", ov::util::Path(u8"./local/file.txt").u32string());
    EXPECT_EQ(U"~/local/file.txt", ov::util::Path(u8"~/local/file.txt").u32string());
    EXPECT_EQ(U"/usr/local/file.txt", ov::util::Path(u8"/usr/local/file.txt").u32string());
    EXPECT_EQ(U"C:\\Users\\file.txt", ov::util::Path(u8"C:\\Users\\file.txt").u32string());
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());

    // from char16_t to u32string
    EXPECT_EQ(U"", ov::util::Path(u"").u32string());
    EXPECT_EQ(U"file.txt", ov::util::Path(u"file.txt").u32string());
    EXPECT_EQ(U"./local/file.txt", ov::util::Path(u"./local/file.txt").u32string());
    EXPECT_EQ(U"~/local/file.txt", ov::util::Path(u"~/local/file.txt").u32string());
    EXPECT_EQ(U"/usr/local/file.txt", ov::util::Path(u"/usr/local/file.txt").u32string());
    EXPECT_EQ(U"C:\\Users\\file.txt", ov::util::Path(u"C:\\Users\\file.txt").u32string());
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());

    // from char32_t to u32string
    EXPECT_EQ(U"", ov::util::Path(U"").u32string());
    EXPECT_EQ(U"file.txt", ov::util::Path(U"file.txt").u32string());
    EXPECT_EQ(U"./local/file.txt", ov::util::Path(U"./local/file.txt").u32string());
    EXPECT_EQ(U"~/local/file.txt", ov::util::Path(U"~/local/file.txt").u32string());
    EXPECT_EQ(U"/usr/local/file.txt", ov::util::Path(U"/usr/local/file.txt").u32string());
    EXPECT_EQ(U"C:\\Users\\file.txt", ov::util::Path(U"C:\\Users\\file.txt").u32string());
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());

    // from char, char8_t, char16_t, char32_t to u32string
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());
#endif

#if (!(defined(__GNUC__) && (__GNUC__ < 12 || __GNUC__ == 12 && __GNUC_MINOR__ < 3)) && \
     !(defined(__clang__) && __clang_major__ < 17))
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95048
    // https://stackoverflow.com/questions/58521857/cross-platform-way-to-handle-stdstring-stdwstring-with-stdfilesystempath

    // from wchar_t to char
    EXPECT_STREQ("", ov::util::Path(L"").string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(L"file.txt").string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(L"./local/file.txt").string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(L"~/local/file.txt").string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(L"/usr/local/file.txt").string().c_str());
    EXPECT_STREQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").string().c_str());

    // from wchar_t to wchar_t
    EXPECT_STREQ(L"", ov::util::Path(L"").wstring().c_str());
    EXPECT_STREQ(L"file.txt", ov::util::Path(L"file.txt").wstring().c_str());
    EXPECT_STREQ(L"./local/file.txt", ov::util::Path(L"./local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"~/local/file.txt", ov::util::Path(L"~/local/file.txt").wstring().c_str());
    EXPECT_STREQ(L"/usr/local/file.txt", ov::util::Path(L"/usr/local/file.txt").wstring().c_str());

    // from wchar_t to char8_t
    EXPECT_STREQ("", ov::util::Path(L"").u8string().c_str());
    EXPECT_STREQ("file.txt", ov::util::Path(L"file.txt").u8string().c_str());
    EXPECT_STREQ("./local/file.txt", ov::util::Path(L"./local/file.txt").u8string().c_str());
    EXPECT_STREQ("~/local/file.txt", ov::util::Path(L"~/local/file.txt").u8string().c_str());
    EXPECT_STREQ("/usr/local/file.txt", ov::util::Path(L"/usr/local/file.txt").u8string().c_str());
    EXPECT_STREQ("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u8string().c_str());

    // from wchar_t to char16_t
    EXPECT_EQ(u"", ov::util::Path(L"").u16string());
    EXPECT_EQ(u"file.txt", ov::util::Path(L"file.txt").u16string());
    EXPECT_EQ(u"./local/file.txt", ov::util::Path(L"./local/file.txt").u16string());
    EXPECT_EQ(u"~/local/file.txt", ov::util::Path(L"~/local/file.txt").u16string());
    EXPECT_EQ(u"/usr/local/file.txt", ov::util::Path(L"/usr/local/file.txt").u16string());
    EXPECT_EQ(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u16string());

#    ifndef MSVC
    // error C2280: 'std::u32string std::experimental::filesystem::v1::path::u32string(void) const': attempting to
    // reference a deleted function
    //  from wchar_t to char32_t
    EXPECT_EQ(U"", ov::util::Path(L"").u32string());
    EXPECT_EQ(U"file.txt", ov::util::Path(L"file.txt").u32string());
    EXPECT_EQ(U"./local/file.txt", ov::util::Path(L"./local/file.txt").u32string());
    EXPECT_EQ(U"~/local/file.txt", ov::util::Path(L"~/local/file.txt").u32string());
    EXPECT_EQ(U"/usr/local/file.txt", ov::util::Path(L"/usr/local/file.txt").u32string());
    EXPECT_EQ(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt", ov::util::Path(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").u32string());
#    endif

    // from char, char8_t, char16_t, char32_t to wchar_t
    EXPECT_STREQ(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path("~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").wstring().c_str());
    EXPECT_STREQ(L"/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path(u8"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").wstring().c_str());
    EXPECT_STREQ(L"/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path(u"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").wstring().c_str());
    EXPECT_STREQ(L"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt",
                 ov::util::Path(U"~/狗/ǡ୫ԩϗ/にほ/ąę/ど/௸ඊƷ/狗.txt").wstring().c_str());
#endif
}
