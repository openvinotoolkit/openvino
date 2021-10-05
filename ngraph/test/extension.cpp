// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"

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
