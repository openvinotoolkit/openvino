// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/unicode_utils.hpp"

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

}  // namespace utils

std::filesystem::path UnicodePathTest::get_path_param() const {
    return std::visit(
        [](const auto& p) {
            // Use OV util to hide some platform details with path creation
            return ov::util::make_path(p);
        },
        GetParam());
}

INSTANTIATE_TEST_SUITE_P(string_paths, UnicodePathTest, testing::Values("test_encoder/test_encoder.encrypted/"));

INSTANTIATE_TEST_SUITE_P(u16_paths, UnicodePathTest, testing::Values(u"test_encoder/dot.folder"));

INSTANTIATE_TEST_SUITE_P(u32_paths, UnicodePathTest, testing::Values(U"test_encoder/dot.folder"));

INSTANTIATE_TEST_SUITE_P(wstring_paths, UnicodePathTest, testing::Values(L"test_encoder/test_encoder.encrypted"));

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
INSTANTIATE_TEST_SUITE_P(unicode_paths,
                         UnicodePathTest,
                         testing::Values("这是.folder", L"这是_folder", u"这是_folder", U"这是_folder"));
#endif

}  // namespace ov::test
