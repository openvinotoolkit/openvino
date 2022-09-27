// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FILE_UTILS_CPP
#define FILE_UTILS_CPP

#include <cstring>
#include <fstream>
#include <string>

#ifdef __MACH__
#    include <mach/clock.h>
#    include <mach/mach.h>
#endif

#include <file_utils.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "ie_common.h"
#include "openvino/util/file_util.hpp"

#ifndef _WIN32
#    include <dlfcn.h>
#    include <limits.h>
#    include <unistd.h>
#else
#    if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#        error "Only WINAPI_PARTITION_DESKTOP is supported, because of GetModuleHandleEx[A|W]"
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <Windows.h>
#endif

long long FileUtils::fileSize(const char* charfilepath) {
    return ov::util::file_size(charfilepath);
}

std::string FileUtils::absoluteFilePath(const std::string& filePath) {
    return ov::util::get_absolute_file_path(filePath);
}

bool FileUtils::directoryExists(const std::string& path) {
    return ov::util::directory_exists(path);
}

void FileUtils::createDirectoryRecursive(const std::string& dirPath) {
    ov::util::create_directory_recursive(dirPath);
}

namespace InferenceEngine {

namespace {

template <typename C, typename = FileUtils::enableIfSupportedChar<C>>
std::basic_string<C> getPathName(const std::basic_string<C>& s) {
    size_t i = s.rfind(ov::util::FileTraits<C>::file_separator, s.length());
    if (i != std::string::npos) {
        return (s.substr(0, i));
    }

    return {};
}

}  // namespace

static std::string getIELibraryPathA() {
#ifdef _WIN32
    CHAR ie_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPSTR>(getIELibraryPath),
                            &hm)) {
        IE_THROW() << "GetModuleHandle returned " << GetLastError();
    }
    GetModuleFileNameA(hm, (LPSTR)ie_library_path, sizeof(ie_library_path));
    return getPathName(std::string(ie_library_path));
#elif defined(__APPLE__) || defined(__linux__)
#    ifdef USE_STATIC_IE
#        ifdef __APPLE__
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    std::string path = getPathName(std::string(info.dli_fname));
#        else
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    std::string path = getPathName(std::string(result, (count > 0) ? count : 0));
#        endif  // __APPLE__
    return FileUtils::makePath(path, std::string("lib"));
#    else
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    std::string path = FileUtils::absoluteFilePath(info.dli_fname);
    return getPathName(path);
#    endif  // USE_STATIC_IE
#else
#    error "Unsupported OS"
#endif  // _WIN32
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::wstring getIELibraryPathW() {
#    ifdef _WIN32
    WCHAR ie_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(getIELibraryPath),
                            &hm)) {
        IE_THROW() << "GetModuleHandle returned " << GetLastError();
    }
    GetModuleFileNameW(hm, (LPWSTR)ie_library_path, sizeof(ie_library_path) / sizeof(ie_library_path[0]));
    return getPathName(std::wstring(ie_library_path));
#    elif defined(__linux__) || defined(__APPLE__)
    return ::ov::util::string_to_wstring(getIELibraryPathA().c_str());
#    else
#        error "Unsupported OS"
#    endif
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::string getIELibraryPath() {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    return ov::util::wstring_to_string(getIELibraryPathW());
#else
    return getIELibraryPathA();
#endif
}

}  // namespace InferenceEngine

#endif
