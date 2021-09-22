// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>

#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"

std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>({}, std::string("template_ov_extension") + IE_BUILD_POSTFIX);
}

TEST(extension, load_extension) {
    ASSERT_NO_THROW(ov::load_extension(get_extension_path()));
}
TEST(extension, load_extension_and_cast) {
    auto extensions = ov::load_extension(get_extension_path());
    ASSERT_EQ(1, extensions.size());
    ASSERT_NE(nullptr, ov::as_type<ov::BaseOpExtension>(extensions[0].get()));
    ASSERT_NE(nullptr, ov::as_type<ov::BaseOpExtension*>(extensions[0].get()));
    ASSERT_NE(nullptr, ov::as_type_ptr<ov::BaseOpExtension>(extensions[0]));
    extensions.clear();
}
