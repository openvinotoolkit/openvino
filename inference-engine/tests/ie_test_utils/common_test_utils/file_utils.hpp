// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "test_constants.hpp"

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

inline void removeIRFiles(const std::string &xmlFilePath, const std::string &binFileName) {
    if (fileExists(xmlFilePath)) {
        std::remove(xmlFilePath.c_str());
    }

    if (fileExists(binFileName)) {
        std::remove(binFileName.c_str());
    }
}
}  // namespace CommonTestUtils
