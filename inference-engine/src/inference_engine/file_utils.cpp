// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include "details/ie_exception.hpp"
#include <fstream>

#include <w_unistd.h>

#ifdef __MACH__
    #include <mach/clock.h>
    #include <mach/mach.h>
#endif

#if defined(WIN32) || defined(WIN64)
    // Copied from linux libc sys/stat.h:
    #define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
    #define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

long long FileUtils::fileSize(const char *fileName) {
    std::ifstream in(fileName, std::ios_base::binary | std::ios_base::ate);
    return in.tellg();
}

void FileUtils::readAllFile(const std::string &file_name, void *buffer, size_t maxSize) {
    std::ifstream inputFile;

    inputFile.open(file_name, std::ios::binary | std::ios::in);
    if (!inputFile.is_open()) THROW_IE_EXCEPTION << "cannot open file " << file_name;
    if (!inputFile.read(reinterpret_cast<char *>(buffer), maxSize)) {
        inputFile.close();
        THROW_IE_EXCEPTION << "cannot read " << maxSize << " bytes from file " << file_name;
    }

    inputFile.close();
}

std::string FileUtils::folderOf(const std::string &filepath) {
    auto pos = filepath.rfind(FileSeparator);
    if (pos == std::string::npos) pos = filepath.rfind(FileSeparator2);
    if (pos == std::string::npos) return "";
    return filepath.substr(0, pos);
}

std::string FileUtils::makePath(const std::string &folder, const std::string &file) {
    if (folder.empty()) return file;
    return folder + FileSeparator + file;
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

bool FileUtils::isSharedLibrary(const std::string& fileName) {
    return 0 == strncasecmp(fileExt(fileName).c_str(), SharedLibraryExt, strlen(SharedLibraryExt));
}
