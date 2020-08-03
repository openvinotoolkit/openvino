// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <fstream>
#include <string>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#ifndef _WIN32
# include <limits.h>
# include <unistd.h>
# include <dlfcn.h>
# include <codecvt>
# include <locale>
#else
# include <Windows.h>
#endif

#include <file_utils.h>
#include <details/ie_so_pointer.hpp>
#include <details/os/os_filesystem.hpp>
#include <details/ie_exception.hpp>

#ifdef ENABLE_UNICODE_PATH_SUPPORT

std::string InferenceEngine::details::wStringtoMBCSstringChar(const std::wstring& wstr) {
#ifdef _WIN32
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);  // NOLINT
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);  // NOLINT
    return strTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
    return wstring_decoder.to_bytes(wstr);
#endif
}

std::wstring InferenceEngine::details::multiByteCharToWString(const char* str) {
#ifdef _WIN32
    int strSize = static_cast<int>(std::strlen(str));
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str, strSize, NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, strSize, &wstrTo[0], size_needed);
    return wstrTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
#endif
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

long long FileUtils::fileSize(const char* charfilepath) {
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = InferenceEngine::details::multiByteCharToWString(charfilepath);
    const wchar_t* fileName = widefilename.c_str();
#else
    const char* fileName = charfilepath;
#endif
    std::ifstream in(fileName, std::ios_base::binary | std::ios_base::ate);
    return in.tellg();
}

void FileUtils::readAllFile(const std::string& string_file_name, void* buffer, size_t maxSize) {
    std::ifstream inputFile;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring file_name = InferenceEngine::details::multiByteCharToWString(string_file_name.c_str());
#else
    std::string file_name = string_file_name;
#endif

    inputFile.open(file_name, std::ios::binary | std::ios::in);
    if (!inputFile.is_open()) THROW_IE_EXCEPTION << "cannot open file " << string_file_name;
    if (!inputFile.read(reinterpret_cast<char*>(buffer), maxSize)) {
        inputFile.close();
        THROW_IE_EXCEPTION << "cannot read " << maxSize << " bytes from file " << string_file_name;
    }

    inputFile.close();
}

namespace InferenceEngine {

namespace {

template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C> >
std::basic_string<C> getPathName(const std::basic_string<C>& s) {
    size_t i = s.rfind(FileUtils::FileTraits<C>::FileSeparator, s.length());
    if (i != std::string::npos) {
        return (s.substr(0, i));
    }

    return {};
}

}  // namespace

static std::string getIELibraryPathA() {
#ifdef _WIN32
    char ie_library_path[4096];
    HMODULE hm = NULL;
    if (!GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)getIELibraryPath, &hm)) {
        THROW_IE_EXCEPTION << "GetModuleHandle returned " << GetLastError();
    }
    GetModuleFileName(hm, (LPSTR)ie_library_path, sizeof(ie_library_path));
    return getPathName(std::string(ie_library_path));
#else
#ifdef USE_STATIC_IE
#ifdef __APPLE__
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    std::string path = getPathName(std::string(info.dli_fname)).c_str();
#else
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    std::string path = getPathName(std::string(result, (count > 0) ? count : 0));
#endif  // __APPLE__
    return FileUtils::makePath(path, std::string( "lib"));
#else
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    return getPathName(std::string(info.dli_fname)).c_str();
#endif  // USE_STATIC_IE
#endif  // _WIN32
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

std::wstring getIELibraryPathW() {
#if defined(_WIN32) || defined(_WIN64)
    wchar_t ie_library_path[4096];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)getIELibraryPath, &hm)) {
        THROW_IE_EXCEPTION << "GetModuleHandle returned " << GetLastError();
    }
    GetModuleFileNameW(hm, (LPWSTR)ie_library_path, sizeof(ie_library_path));
    return getPathName(std::wstring(ie_library_path));
#else
    return details::multiByteCharToWString(getIELibraryPathA().c_str());
#endif
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

std::string getIELibraryPath() {
#ifdef ENABLE_UNICODE_PATH_SUPPORT
    return details::wStringtoMBCSstringChar(getIELibraryPathW());
#else
    return getIELibraryPathA();
#endif
}

}  // namespace InferenceEngine
