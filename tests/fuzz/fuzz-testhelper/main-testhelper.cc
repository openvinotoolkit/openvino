// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/*!
\file
\brief Replacement of libFuzzer main entrypoint.

[libFuzzer](https://llvm.org/docs/LibFuzzer.html), part of LLVM toolchain,
implements `main` entry point which runs in-process fuzzing. This provides
a simplified `main` entry point implementation which is limited to processing
the inputs.
*/

#if !defined(WITH_LIBFUZZER)

#include <stdint.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
# include <windows.h>
#else  // WIN32
# include <dirent.h>
# include <sys/stat.h>
# include <sys/types.h>
#endif  // WIN32

/// Fuzzing target
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

/// Get basename from path
std::string basename(std::string const& path) {
    std::string str = std::string("/") + path;
    return str.substr(str.find_last_of("/\\") + 1);
}

/// Get directory content
std::vector<std::string> list_dir(std::string const& path) {
    std::vector<std::string> res;
#ifdef _WIN32
    WIN32_FIND_DATA find_data;
    HANDLE find_handle;
    find_handle = FindFirstFile((path + "\\*").c_str(), &find_data);
    if (INVALID_HANDLE_VALUE != find_handle) {
        do {
            std::string filename(find_data.cFileName);
            if (filename == "." || filename == "..") continue;
            res.push_back(path + "\\" + filename);
        } while (FindNextFile(find_handle, &find_data));
        FindClose(find_handle);
    }
#else   // WIN32
    DIR* dir = opendir(path.c_str());
    if (dir) {
        struct dirent* entry;
        while (NULL != (entry = readdir(dir))) {
            if (DT_REG == entry->d_type) res.push_back(path + "/" + std::string(entry->d_name));
        }
        closedir(dir);
        dir = NULL;
    }
#endif  // WIN32
    return res;
}

// Check if file by given path is a directory.
bool is_dir(std::string const& path) {
#ifdef _WIN32
    return 0 != (FILE_ATTRIBUTE_DIRECTORY & GetFileAttributes(path.c_str()));
#else   // WIN32
    struct stat stat_res = {0};
    stat(path.c_str(), &stat_res);
    return S_IFDIR & stat_res.st_mode;
#endif  // WIN32
}

// Print usage help
void print_usage(const std::string& program_name, std::ostream* os) {
    *os << "Usage: " << program_name << " INPUT" << std::endl;
}

/// Main entrypoint
extern "C" int main(int argc, char* argv[]) {
    std::string program_name = basename(argv[0]);

    // Parse command line options
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        // Ignore all options but positional arguments
        if ('-' == arg[0]) {
            std::cout << "Ignoring option " << arg << std::endl;
            continue;
        }
        positional.push_back(arg);
    }
    if (1 != positional.size()) {
        std::cerr << program_name << ": error: wrong number of positional arguments." << std::endl;
        print_usage(program_name, &std::cerr);
        return -1;
    }

    // Run input files through test function
    std::vector<std::string> input_files;
    if (is_dir(positional[0])) {
        std::cout << "Loading corpus dir: " << positional[0] << std::endl;
        input_files = list_dir(positional[0]);
    } else {
        std::cout << "Running: " << positional[0] << std::endl;
        input_files.push_back(positional[0]);
    }
    time_t time_total = 0;
    for (auto const& path : input_files) {
        std::ifstream test_file(path, std::ios::binary);
        if (!test_file) {
            std::cerr << program_name << ": error: failed to open \"" << path << "\"" << std::endl;
            return -2;
        }
        std::ostringstream data;
        data << test_file.rdbuf();
        test_file.close();

        time_t time_start = time(nullptr);
        int fuzzer_res;
        if (0 != (fuzzer_res = LLVMFuzzerTestOneInput((const uint8_t*)data.str().c_str(), data.str().size()))) {
            std::cerr << program_name << ": error: testing \"" << path << "\" fails" << std::endl;
            return fuzzer_res;
        }
        time_total += time(nullptr) - time_start;
    }
    std::cout << "Executed " << input_files.size() << " item(s) in " << time_total << " seconds" << std::endl;
    return 0;
}

#endif  // !defined(WITH_LIBFUZZER)
