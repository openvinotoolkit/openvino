// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>

#include "openvino/core/op_extension.hpp"
#include "openvino/core/utils/file_utils.hpp"

std::string get_extension_path() {
    return ov::utils::make_plugin_library_name<char>({}, std::string("template_ov_extension") + IE_BUILD_POSTFIX);
}

// TEST(extension, load_extension) {
//     ASSERT_NO_THROW(ov::load_extensios(get_extension_path()));
// }
TEST(extension, load_extension_and_cast) {
    auto extensions = ov::load_extension(get_extension_path());
    ASSERT_EQ(2, extensions.size());
    ASSERT_NE(nullptr, dynamic_cast<ov::OpExtension*>(extensions[0].get()));
    ASSERT_NE(nullptr, std::dynamic_pointer_cast<ov::OpExtension>(extensions[1]));
    extensions.clear();
    std::cout << "AAAA " << std::endl;
}
