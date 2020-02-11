// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>

#include <cstring>
#include <fstream>
#include <string>

#include "details/ie_exception.hpp"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include "details/os/os_filesystem.hpp"

#if defined(WIN32) || defined(WIN64)
// Copied from linux libc sys/stat.h:
#define S_ISREG(m) (((m)&S_IFMT) == S_IFREG)
#define S_ISDIR(m) (((m)&S_IFMT) == S_IFDIR)
#endif

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
