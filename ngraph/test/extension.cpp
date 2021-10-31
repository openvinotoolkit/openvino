// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"
#include "so_extension.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#        error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibrary[A|W]"
#    endif
#elif defined(__linux) || defined(__APPLE__)
#    include <dlfcn.h>
#endif

static std::string find_my_pathname() {
#ifdef _WIN32
    HMODULE hModule = GetModuleHandleW(SHARED_LIB_PREFIX L"ngraph" SHARED_LIB_SUFFIX);
    WCHAR wpath[MAX_PATH];
    GetModuleFileNameW(hModule, wpath, MAX_PATH);
    std::wstring ws(wpath);
    std::string path(ws.begin(), ws.end());
    replace(path.begin(), path.end(), '\\', '/');
    path = ov::util::get_directory(path);
    path += "/";
    return path;
#elif defined(__linux) || defined(__APPLE__)
    Dl_info dl_info;
    dladdr(reinterpret_cast<void*>(ov::replace_output_update_name), &dl_info);
    return ov::util::get_directory(dl_info.dli_fname);
#else
#    error "Unsupported OS"
#endif
}
std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>(find_my_pathname(),
                                                    std::string("template_ov_extension") + IE_BUILD_POSTFIX);
}

TEST(extension, load_extension) {
    EXPECT_NO_THROW(ov::detail::load_extensions(get_extension_path()));
}

TEST(extension, load_extension_and_cast) {
    std::vector<ov::Extension::Ptr> so_extensions;
    EXPECT_NO_THROW(so_extensions = ov::detail::load_extensions(get_extension_path()));
    EXPECT_EQ(1, so_extensions.size());
    std::vector<ov::Extension::Ptr> extensions;
    std::vector<std::shared_ptr<void>> so;
    for (const auto& ext : so_extensions) {
        if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext)) {
            extensions.emplace_back(so_ext->extension());
            so.emplace_back(so_ext->shared_object());
        }
    }
    so_extensions.clear();
    EXPECT_EQ(1, extensions.size());
    EXPECT_NE(nullptr, dynamic_cast<ov::BaseOpExtension*>(extensions[0].get()));
    EXPECT_NE(nullptr, std::dynamic_pointer_cast<ov::BaseOpExtension>(extensions[0]));
    extensions.clear();
}
