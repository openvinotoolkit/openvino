// Copyright (C) 2018-2023 Intel Corporation
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
    return ov::util::get_directory(path);
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
    return ov::util::path_join({getExecutableDirectory(), relModelPath});
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
                                  [&separator](std::string& a, const std::string&) -> std::string {
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
                              [&separator](std::string& a, const std::string& b) -> std::string {
                                  return a.empty() ? a += b : a += separator + b;
                              });
    return output;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
