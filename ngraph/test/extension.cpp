// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>

#include "load_extensions.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"

static std::string find_my_pathname() {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    return ov::util::wstring_to_string(ov::util::get_ov_library_path());
#else
    return ov::util::get_ov_library_path();
#endif
}
std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>(find_my_pathname(),
                                                    std::string("template_ov_extension") + IE_BUILD_POSTFIX);
}

TEST(extension, load_extension) {
    std::vector<ov::Extension::Ptr> extensions;
    EXPECT_NO_THROW(extensions = ov::detail::load_extensions(get_extension_path()));
    EXPECT_NO_THROW(ov::detail::unload_extensions(extensions));
}

TEST(extension, load_extension_and_cast) {
    std::vector<ov::Extension::Ptr> extensions;
    EXPECT_NO_THROW(extensions = ov::detail::load_extensions(get_extension_path()));
    EXPECT_EQ(1, extensions.size());
    EXPECT_NE(nullptr, dynamic_cast<ov::BaseOpExtension*>(extensions[0].get()));
    EXPECT_NE(nullptr, std::dynamic_pointer_cast<ov::BaseOpExtension>(extensions[0]));
    EXPECT_NO_THROW(ov::detail::unload_extensions(extensions));
    extensions.clear();
}
