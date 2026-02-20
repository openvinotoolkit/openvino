// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/unicode_utils.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

namespace ov {
namespace test {
namespace utils {

const std::vector<std::wstring> test_unicode_postfix_vector = {L"unicode_Яㅎあ",
                                                               L"ひらがな日本語",
                                                               L"大家有天分",
                                                               L"עפצקרשתםןףץ",
                                                               L"ث خ ذ ض ظ غ",
                                                               L"그것이정당하다",
                                                               L"АБВГДЕЁЖЗИЙ",
                                                               L"СТУФХЦЧШЩЬЮЯ"};

}  // namespace utils
}  // namespace test
}  // namespace ov

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

namespace ov::test {

std::filesystem::path UnicodePathTest::get_path_param() const {
    return std::visit(
        [](const auto& p) {
            // Use OV util to hide some platform details with path creation
            return ov::util::make_path(p);
        },
        GetParam());
}

INSTANTIATE_TEST_SUITE_P(string_paths, UnicodePathTest, testing::Values("test_folder"));
INSTANTIATE_TEST_SUITE_P(u16_paths, UnicodePathTest, testing::Values(u"test_folder"));
INSTANTIATE_TEST_SUITE_P(u32_paths, UnicodePathTest, testing::Values(U"test_folder"));
INSTANTIATE_TEST_SUITE_P(wstring_paths, UnicodePathTest, testing::Values(L"test_folder"));

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
INSTANTIATE_TEST_SUITE_P(unicode_paths, UnicodePathTest, unicode_paths);
#endif

}  // namespace ov::test
