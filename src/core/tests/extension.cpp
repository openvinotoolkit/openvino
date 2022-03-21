// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"
#include "so_extension.hpp"

inline std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>({}, std::string("openvino_template_extension") + IE_BUILD_POSTFIX);
}

TEST(extension, load_extension) {
    EXPECT_NO_THROW(ov::detail::load_extensions(get_extension_path()));
}

TEST(extension, load_extension_and_cast) {
    std::vector<ov::Extension::Ptr> so_extensions = ov::detail::load_extensions(get_extension_path());
    ASSERT_LE(1, so_extensions.size());
    std::vector<ov::Extension::Ptr> extensions;
    std::vector<std::shared_ptr<void>> so;
    for (const auto& ext : so_extensions) {
        if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext)) {
            extensions.emplace_back(so_ext->extension());
            so.emplace_back(so_ext->shared_object());
        }
    }
    so_extensions.clear();
    EXPECT_LE(1, extensions.size());
    EXPECT_NE(nullptr, dynamic_cast<ov::BaseOpExtension*>(extensions[0].get()));
    EXPECT_NE(nullptr, std::dynamic_pointer_cast<ov::BaseOpExtension>(extensions[0]));
    extensions.clear();
}
