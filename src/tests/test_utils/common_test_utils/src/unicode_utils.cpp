// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/unicode_utils.hpp"

namespace {

template <typename CharT>
std::basic_string<CharT> convert_ascii_string(const std::string& value) {
    std::basic_string<CharT> converted;
    converted.reserve(value.size());
    for (unsigned char ch : value) {
        converted.push_back(static_cast<CharT>(ch));
    }
    return converted;
}

}  // namespace

namespace ov::test {

namespace utils {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

const std::vector<std::wstring> test_unicode_postfix_vector = {L"unicode_Яㅎあ",
                                                               L"ひらがな日本語",
                                                               L"大家有天分",
                                                               L"עפצקרשתםןףץ",
                                                               L"ث خ ذ ض ظ غ",
                                                               L"그것이정당하다",
                                                               L"АБВГДЕЁЖЗИЙ",
                                                               L"СТУФХЦЧШЩЬЮЯ"};

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

template <>
std::string cast_string_to_type<std::string>(const std::string& value) {
    return value;
}

template <>
std::wstring cast_string_to_type<std::wstring>(const std::string& value) {
    return ov::util::string_to_wstring(value);
}

template <>
std::u16string cast_string_to_type<std::u16string>(const std::string& value) {
    return convert_ascii_string<char16_t>(value);
}

template <>
std::u32string cast_string_to_type<std::u32string>(const std::string& value) {
    return convert_ascii_string<char32_t>(value);
}

}  // namespace utils

std::filesystem::path UnicodePathTest::get_path_param() const {
    return std::visit(
        [](const auto& p) {
            // Use OV util to hide some platform details with path creation
            return ov::util::make_path(p);
        },
        GetParam());
}
std::filesystem::path UnicodePathTest::fs_path_from_variant() const {
    return std::visit(
        [](const auto& p) {
            return std::filesystem::path(p);
        },
        GetParam());
}

INSTANTIATE_TEST_SUITE_P(string_paths,
                         UnicodePathTest,
                         testing::Values("test_encoder/test_encoder.encrypted/",
                                         "test_encoder/test_encoder.encrypted"));

INSTANTIATE_TEST_SUITE_P(u16_paths,
                         UnicodePathTest,
                         testing::Values(u"test_encoder/dot.folder", u"test_encoder/dot.folder/"));

INSTANTIATE_TEST_SUITE_P(u32_paths,
                         UnicodePathTest,
                         testing::Values(U"test_encoder/dot.folder", U"test_encoder/dot.folder/"));

INSTANTIATE_TEST_SUITE_P(wstring_paths,
                         UnicodePathTest,
                         testing::Values(L"test_encoder/test_encoder.encrypted",
                                         L"test_encoder/test_encoder.encrypted/"));

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
INSTANTIATE_TEST_SUITE_P(
    unicode_paths,
    UnicodePathTest,
    testing::Values("这是.folder", L"这是_folder", L"这是_folder/", u"这是_folder/", U"这是_folder/"));
#endif

}  // namespace ov::test
