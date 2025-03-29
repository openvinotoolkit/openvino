// Copyright (C) 2018-2025 Intel Corporation
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
