// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tests_file_utils.hpp>
#include <fstream>
#include <string>

#include <sys/stat.h>

#include "common_test_utils/file_utils.hpp"

#ifdef __MACH__
# include <mach/clock.h>
# include <mach/mach.h>
#endif

#ifdef _WIN32
// Copied from linux libc sys/stat.h:
#ifndef S_ISREG
# define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#endif
#ifndef S_ISDIR
# define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif
#endif

using namespace ::testing;
using namespace std;

void FileUtils::readAllFile(const std::string &file_name, void *buffer, size_t maxSize) {
    std::ifstream inputFile;
    inputFile.open(file_name, std::ios::binary | std::ios::in);
    if (!inputFile.is_open()) OPENVINO_THROW("cannot open file ", file_name);
    if (!inputFile.read(static_cast<char *> (buffer), maxSize)) {
        inputFile.close();
        OPENVINO_THROW("cannot read ", maxSize, " bytes from file ", file_name);
    }

    inputFile.close();
}

std::string FileUtils::folderOf(const std::string &filepath) {
    auto pos = filepath.rfind(ov::test::utils::FileSeparator);
    if (pos == std::string::npos) pos = filepath.rfind(ov::test::utils::FileSeparator);
    if (pos == std::string::npos) return "";
    return filepath.substr(0, pos);
}

std::string FileUtils::fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

std::string FileUtils::fileExt(const char *filename) {
    return fileExt(std::string(filename));
}

std::string FileUtils::fileExt(const std::string &filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

