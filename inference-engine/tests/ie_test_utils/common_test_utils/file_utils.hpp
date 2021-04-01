// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "test_constants.hpp"
#include "w_dirent.h"
#include "common_utils.hpp"

#ifdef _WIN32
#include <direct.h>
#define rmdir(dir) _rmdir(dir)
#else  // _WIN32
#include <unistd.h>
#endif  // _WIN32

namespace CommonTestUtils {

template<class T>
inline std::string to_string_c_locale(T value) {
    std::stringstream val_stream;
    val_stream.imbue(std::locale("C"));
    val_stream << value;
    return val_stream.str();
}

inline std::string makePath(const std::string &folder, const std::string &file) {
    if (folder.empty()) return file;
    return folder + FileSeparator + file;
}

inline long long fileSize(const char *fileName) {
    std::ifstream in(fileName, std::ios_base::binary | std::ios_base::ate);
    return in.tellg();
}

inline long long fileSize(const std::string &fileName) {
    return fileSize(fileName.c_str());
}

inline bool fileExists(const char *fileName) {
    return fileSize(fileName) >= 0;
}

inline bool fileExists(const std::string &fileName) {
    return fileExists(fileName.c_str());
}

inline void createFile(const std::string& filename, const std::string& content) {
    std::ofstream outfile(filename);
    outfile << content;
    outfile.close();
}

inline void removeFile(const std::string& path) {
    if (!path.empty()) {
        remove(path.c_str());
    }
}

inline void removeIRFiles(const std::string &xmlFilePath, const std::string &binFileName) {
    if (fileExists(xmlFilePath)) {
        std::remove(xmlFilePath.c_str());
    }

    if (fileExists(binFileName)) {
        std::remove(binFileName.c_str());
    }
}

// Removes all files with extension=ext from the given directory
// Return value:
// < 0 - error
// >= 0 - count of removed files
inline int removeFilesWithExt(std::string path, std::string ext) {
    struct dirent *ent;
    DIR *dir = opendir(path.c_str());
    int ret = 0;
    if (dir != nullptr) {
        while ((ent = readdir(dir)) != NULL) {
            auto file = makePath(path, std::string(ent->d_name));
            struct stat stat_path;
            stat(file.c_str(), &stat_path);
            if (!S_ISDIR(stat_path.st_mode) && endsWith(file, "." + ext)) {
                auto err = std::remove(file.c_str());
                if (err != 0) {
                    closedir(dir);
                    return err;
                }
                ret++;
            }
        }
        closedir(dir);
    }

    return ret;
}

inline int removeDir(const std::string &path) {
    return rmdir(path.c_str());
}

inline bool directoryExists(const std::string &path) {
    struct stat sb;

    if (stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        return true;
    }

    return false;
}

}  // namespace CommonTestUtils
