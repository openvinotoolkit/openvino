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

const static auto basic_paths = testing::Values("test_encoder/test_encoder.encrypted",
                                                L"test_encoder/test_encoder.encrypted",
                                                u"test_encoder/dot.folder",
                                                U"test_encoder/dot.folder");

const static auto basic_paths_dir = testing::Values("test_encoder/test_encoder.encrypted/",
                                                    L"test_encoder/test_encoder.encrypted/",
                                                    u"test_encoder/dot.folder/",
                                                    U"test_encoder/dot.folder/");

INSTANTIATE_TEST_SUITE_P(basic_paths, UnicodePathTest, basic_paths);
INSTANTIATE_TEST_CASE_P(basic_paths, FileUtilTestP, basic_paths);
INSTANTIATE_TEST_CASE_P(basic_paths_dir, FileUtilTestP, basic_paths_dir);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
const static auto unicode_paths = testing::Values("这是.folder", L"这是_folder", u"这是_folder", U"这是_folder");
const static auto unicode_paths_dir =
    testing::Values("这是.folder/", L"这是_folder/", u"这是_folder/", U"这是_folder/");

INSTANTIATE_TEST_SUITE_P(unicode_paths, UnicodePathTest, unicode_paths);
INSTANTIATE_TEST_SUITE_P(unicode_paths, FileUtilTestP, unicode_paths);
INSTANTIATE_TEST_SUITE_P(unicode_paths_dir, FileUtilTestP, unicode_paths_dir);
#endif

}  // namespace ov::test
