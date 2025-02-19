// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"

#include "precomp.hpp"

#ifdef __APPLE__
#    include <mach-o/dyld.h>
#endif

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <direct.h>
#    include <stdlib.h>
#    include <windows.h>
#else
#    include <dlfcn.h>
#    include <limits.h>
#    include <unistd.h>
#endif

namespace ov {
namespace test {
namespace utils {

namespace {

template <typename C,
          typename = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type>
std::basic_string<C> get_path_name(const std::basic_string<C>& s) {
    size_t i = s.rfind(ov::util::FileTraits<C>::file_separator, s.length());
    if (i != std::string::npos) {
        return (s.substr(0, i));
    }

    return {};
}

#if defined __GNUC__ || defined __clang__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-function"
#endif

std::string getOpenvinoLibDirectoryA() {
#ifdef _WIN32
    CHAR ov_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPSTR>(ov::get_openvino_version),
                            &hm)) {
        std::stringstream ss;
        ss << "GetModuleHandle returned " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    GetModuleFileNameA(hm, (LPSTR)ov_library_path, sizeof(ov_library_path));
    return get_path_name(std::string(ov_library_path));
#elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(ov::get_openvino_version), &info);
    return get_path_name(ov::util::get_absolute_file_path(info.dli_fname)).c_str();
#else
#    error "Unsupported OS"
#endif  // _WIN32
}

#if defined __GNUC__ || defined __clang__
#    pragma GCC diagnostic pop
#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::wstring getOpenvinoLibDirectoryW() {
#    ifdef _WIN32
    WCHAR ov_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(ov::get_openvino_version),
                            &hm)) {
        std::stringstream ss;
        ss << "GetModuleHandle returned " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    GetModuleFileNameW(hm, (LPWSTR)ov_library_path, sizeof(ov_library_path) / sizeof(ov_library_path[0]));
    return get_path_name(std::wstring(ov_library_path));
#    elif defined(__linux__) || defined(__APPLE__) || defined(__EMSCRIPTEN__)
    return ov::util::string_to_wstring(getOpenvinoLibDirectoryA());
#    else
#        error "Unsupported OS"
#    endif
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

}  // namespace

std::string getOpenvinoLibDirectory() {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    return ov::util::wstring_to_string(getOpenvinoLibDirectoryW());
#else
    return getOpenvinoLibDirectoryA();
#endif
}

std::string getExecutableDirectory() {
    std::string path;
#ifdef _WIN32
    char buffer[MAX_PATH];
    int len = GetModuleFileNameA(NULL, buffer, MAX_PATH);
#elif defined(__APPLE__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getExecutableDirectory), &info);
    const char* buffer = info.dli_fname;
    int len = std::strlen(buffer);
#else
    char buffer[PATH_MAX];
    int len = readlink("/proc/self/exe", buffer, PATH_MAX);
#endif
    if (len < 0) {
        throw "Can't get test executable path name";
    }
    path = std::string(buffer, len);
    return ov::util::get_directory(path).string();
}

std::string getCurrentWorkingDir() {
    std::string path;
#ifdef _WIN32
    char* buffer = _getcwd(NULL, 0);
    if (buffer != NULL) {
        path = std::string(buffer);
        free(buffer);
    }
#else
    char buffer[PATH_MAX];
    auto result = getcwd(buffer, sizeof(buffer));
    if (result != NULL) {
        path = std::string(buffer);
    } else {
        int error = errno;
        std::ostringstream str;
        str << "Can't get access to the current working directory, error:" << error;
        throw std::runtime_error(str.str());
    }
#endif
    return path;
}

std::string getModelFromTestModelZoo(const std::string& relModelPath) {
    return ov::util::path_join({getExecutableDirectory(), relModelPath}).string();
}

std::string getRelativePath(const std::string& from, const std::string& to) {
    auto split_path = [](const std::string& path) -> std::vector<std::string> {
        std::string sep{ov::util::FileTraits<char>::file_separator};
        std::vector<std::string> retvalue;
        size_t start = 0;
        size_t end = 0;
        std::string token;
        while ((end = path.find(sep, start)) != std::string::npos) {
            token = path.substr(start, end - start);
            start = end + 1;
            if (!token.empty())
                retvalue.push_back(token);
        }

        token = path.substr(start);
        if (!token.empty()) {
            retvalue.push_back(token);
        }
        return retvalue;
    };

    auto from_vec = split_path(from);
    auto to_vec = split_path(to);

    auto mismatch_it = std::mismatch(from_vec.begin(), from_vec.end(), to_vec.begin());
    if (mismatch_it.first == from_vec.end() && mismatch_it.second == to_vec.end()) {
        return {};
    }

    std::string separator(1, ov::util::FileTraits<char>::file_separator);
    std::string output;
    //  generates path to the top common directory from the start directory
    if (mismatch_it.first != from_vec.end()) {
        // adds signs: "../" until it meets the top common directory
        // for example if start path is: /aaa/bbb/ddd/eee and destination path is: /aaa/bbb/cc/test_app
        // it generates: "../../"
        output += std::accumulate(mismatch_it.first,
                                  from_vec.end(),
                                  std::string{},
                                  [&separator](std::string a, const std::string&) -> std::string {
                                      return a += ".." + separator;
                                  });
    }
    // adds path to the destination. If before generates path contains signs: "../",
    // for example if start path is: "/aaa/bbb/ddd/eee" and destination path is: "/aaa/bbb/cc/test_app"
    // To the generated path: "../../" adds: "cc/test_app",
    // the output path is: "../../cc/test_app"
    output += std::accumulate(mismatch_it.second,
                              to_vec.end(),
                              std::string{},
                              [&separator](std::string a, const std::string& b) -> std::string {
                                  return a.empty() ? a += b : a += separator + b;
                              });
    return output;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
