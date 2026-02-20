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

const static auto test_paths = testing::Values("test_folder", L"test_folder", u"test_folder", U"test_folder");
INSTANTIATE_TEST_SUITE_P(test_paths, UnicodePathTest, test_paths);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
const static auto unicode_paths = testing::Values("这是_folder", L"这是_folder", u"这是_folder", U"这是_folder");

INSTANTIATE_TEST_SUITE_P(unicode_paths, UnicodePathTest, unicode_paths);
#endif

}  // namespace ov::test
